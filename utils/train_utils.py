import gc
import multiprocessing
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from models import MP_Sampling
from models.Geo2Vec import Geo2Vec_Model, Geo2Vec_Dataset, SDFLoss, identity_collate


def sample_geo2vec_dataset(polys_dict, num_process, samples_perUnit, point_sample, sample_band_width,
                            uniformed_sample_perUnit, training_ratio=0.95):
    """Sample signed-distance training data for `polys_dict` and split it into train/val datasets.

    Kept separate from `train_geo2vec_model` since this part is CPU/multiprocessing-bound
    (polygon sampling) while training is GPU-bound - callers can sample once and reuse the
    resulting datasets across multiple training runs/hyperparameters.

    Returns:
        train_dataset, val_dataset (Geo2Vec_Dataset splits): ready for a DataLoader.
        max_id (int): largest polygon id in `polys_dict`. Ids are 0-indexed, so pass
            `n_poly=max_id + 1` to `Geo2Vec_Model`/`train_geo2vec_model` to size the
            embedding table exactly, with no unused rows.
    """
    multiprocessing.set_start_method("spawn", force=True)
    samples = MP_Sampling.MP_sample(polys_dict, num_process, samples_perUnit=samples_perUnit,
                                     point_sample=point_sample, sample_band_width=sample_band_width,
                                     uniformed_sample_perUnit=uniformed_sample_perUnit)

    max_id = max(polys_dict.keys())
    total_dataset = Geo2Vec_Dataset(samples, polys_dict.keys())
    samples = None
    gc.collect()

    train_size = round(training_ratio * len(total_dataset))
    val_size = len(total_dataset) - train_size
    train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])
    total_dataset = None
    gc.collect()

    return train_dataset, val_dataset, max_id


def train_geo2vec_model(train_dataset, val_dataset, max_id, device, epochs, batch_size,
                         z_size=256, hidden_size=256, num_freqs=16, num_layers=8,
                         code_reg_weight=0.0, weight_decay=0.01, polar_fourier=False, log_sampling=False,
                         lr=0.001, num_workers=0, save_model_path=None, verbose=True):
    """Train a Geo2Vec_Model on datasets produced by `sample_geo2vec_dataset`.

    DataLoader uses num_workers=0 with identity_collate since Geo2Vec_Dataset already
    returns whole batches via __getitems__, and tensors move to `device` with
    non_blocking=True (paired with pin_memory) to overlap transfer with compute.

    Returns:
        model (Geo2Vec_Model): model at its final training epoch (not necessarily the
            epoch with the lowest validation loss - see `embedding`/`save_model_path` for that).
        embedding (np.ndarray | None): poly_embedding_layer weights captured at the epoch
            with the lowest validation loss seen so far, shape [max_id + 1, z_size]. None if
            no epoch ever improved on `best_val_loss` (e.g. epochs=0).
        best_val_loss (float): lowest validation loss observed.
    """
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                             num_workers=num_workers, pin_memory=True, collate_fn=identity_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                 num_workers=num_workers, pin_memory=True, collate_fn=identity_collate)

    if verbose:
        print(f"In average training samples per entity: {len(train_dataset) / (max_id + 1)}")

    # n_poly = max_id + 1 since ids are 0-indexed; using more would leave unused,
    # never-trained rows at the end of the embedding table.
    model = Geo2Vec_Model(n_poly=max_id + 1, z_size=z_size, hidden_size=hidden_size, num_freqs=num_freqs,
                           weight_decay=weight_decay, log_sampling=log_sampling,
                           polar_fourier=polar_fourier, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = SDFLoss(code_reg_weight=code_reg_weight, sum=True)

    best_val_loss = float('inf')
    embedding = None
    for epoch in tqdm(range(epochs), desc='Training...'):
        model.train()
        epoch_loss = 0
        for id, sample, dist in dataloader:
            id = id.to(device, non_blocking=True)
            sample = sample.to(device, non_blocking=True)
            dist = dist.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(id, sample)
            latent_code = model.poly_embedding_layer(id)
            loss = loss_fn(output, dist, latent_code)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        test_epoch_loss = 0
        with torch.no_grad():
            model.eval()
            for id, sample, dist in val_dataloader:
                id = id.to(device, non_blocking=True)
                sample = sample.to(device, non_blocking=True)
                dist = dist.to(device, non_blocking=True)
                output = model(id, sample)
                loss = F.l1_loss(output, dist, reduction='mean')
                test_epoch_loss += loss.item()

        if best_val_loss > test_epoch_loss:
            best_val_loss = test_epoch_loss
            embedding = model.poly_embedding_layer.weight.data.cpu().numpy()
            if save_model_path is not None:
                torch.save(model.state_dict(), save_model_path)

        if verbose:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}, TEST Loss: {test_epoch_loss}')

    return model, embedding, best_val_loss
