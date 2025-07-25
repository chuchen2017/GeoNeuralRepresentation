import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import multiprocessing
import gc
import argparse
import torch
import random
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
# from sys import path
# path.append('..')
import utils.test_representation as test_representation
from models.Geo2Vec import Geo2Vec_Model, Geo2Vec_Dataset, SDFLoss
from models import MP_Sampling
from utils.data_loader import load_data
import utils.visualization as visualization


def get_args():
    parser = argparse.ArgumentParser(description="Geo2vec Training Config")

    # file_path: where the data is stored in pkl or gpkg format
    file_path = r'data\ShapeClassification.gpkg'
    parser.add_argument('--file_path', type=str, default=file_path)
    # Save file path
    save_path = os.path.splitext(file_path)[0] + '.pth'
    parser.add_argument('--save_file_name', type=str, default=save_path)

    # Sampling Parameters
    # For location
    parser.add_argument('--num_process', type=int, default=10)
    parser.add_argument('--samples_perUnit_location', type=int, default=4000)
    parser.add_argument('--point_sample_location', type=int, default=10)
    parser.add_argument('--sample_band_width_location', type=float, default=0.1)
    parser.add_argument('--uniformed_sample_perUnit_location', type=int, default=30)

    # For shape
    parser.add_argument('--samples_perUnit_shape', type=int, default=100)
    parser.add_argument('--point_sample_shape', type=int, default=20)
    parser.add_argument('--sample_band_width_shape', type=float, default=0.1)
    parser.add_argument('--uniformed_sample_perUnit_shape', type=int, default=20)

    # Training parameters
    # For location
    parser.add_argument('--batch_size', type=int, default=1024 * 20)
    parser.add_argument('--num_workers', type=int, default=8)  # Training dataload number of works

    parser.add_argument('--epochs_location', type=int, default=2)
    parser.add_argument('--num_layers_location', type=int, default=8)
    parser.add_argument('--z_size_location', type=int, default=256)
    parser.add_argument('--hidden_size_location', type=int, default=256)
    parser.add_argument('--num_freqs_location', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--code_reg_weight_location', type=float, default=0.0)
    parser.add_argument('--weight_decay_location', type=float, default=0.01)
    parser.add_argument('--polar_fourier_location', action='store_true', default=False)
    parser.add_argument('--log_sampling_location', action='store_true', default=False)
    parser.add_argument('--training_ratio_location', type=float, default=0.95)

    # For shape
    parser.add_argument('--epochs_shape', type=int, default=2)
    parser.add_argument('--num_layers_shape', type=int, default=8)
    parser.add_argument('--z_size_shape', type=int, default=256)
    parser.add_argument('--hidden_size_shape', type=int, default=256)
    parser.add_argument('--num_freqs_shape', type=int, default=8)
    parser.add_argument('--device_shape', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--code_reg_weight_shape', type=float, default=1.0)
    parser.add_argument('--weight_decay_shape', type=float, default=0.01)
    parser.add_argument('--polar_fourier_shape', action='store_true', default=False)
    parser.add_argument('--log_sampling_shape', action='store_true', default=True)
    parser.add_argument('--training_ratio_shape', type=float, default=0.95)

    # Testing options
    # For location
    parser.add_argument('--test_representation_location', type=bool, default=True)
    parser.add_argument('--visualSDF_location', type=bool, default=True)
    # For shape
    parser.add_argument('--test_representation_shape', type=bool, default=True)
    parser.add_argument('--visualSDF_shape', type=bool, default=True)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)

    file_path = args.file_path
    save_file_name = args.save_file_name

    num_workers = args.num_workers
    z_size = args.z_size_location
    hidden_size = args.hidden_size_location
    num_freqs = args.num_freqs_location
    code_reg_weight = args.code_reg_weight_location
    weight_decay = args.weight_decay_location
    num_layers = args.num_layers_location
    epochs = args.epochs_location
    polar_fourier = args.polar_fourier_location
    log_sampling = args.log_sampling_location

    batch_size = args.batch_size
    num_process = args.num_process
    training_ratio = args.training_ratio_location
    samples_perUnit = args.samples_perUnit_location
    point_sample = args.point_sample_location
    sample_band_width = args.sample_band_width_location
    uniformed_sample_perUnit = args.uniformed_sample_perUnit_location

    polys_dict_shape, polys_dict_loc, classification_labels, areas_labels, perimeters_labels, num_edges_labels = load_data(
        file_path)
    multiprocessing.set_start_method("spawn", force=True)
    samples = MP_Sampling.MP_sample(polys_dict_loc, num_process, samples_perUnit=samples_perUnit,
                                    point_sample=point_sample,
                                    sample_band_width=sample_band_width,
                                    uniformed_sample_perUnit=uniformed_sample_perUnit)

    max_id = max(polys_dict_loc.keys())
    total_dataset = Geo2Vec_Dataset(samples, polys_dict_loc.keys())
    samples = None
    gc.collect()

    train_size = round(training_ratio * len(total_dataset))
    val_size = len(total_dataset) - train_size
    train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])
    total_dataset = None
    gc.collect()
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                            num_workers=num_workers,
                            pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                num_workers=num_workers,
                                pin_memory=True)

    print(f"In average training samples per entity: {len(train_dataset) / len(polys_dict_loc)}")

    model = Geo2Vec_Model(n_poly=max_id + 2, z_size=z_size, hidden_size=hidden_size, num_freqs=num_freqs,
                          weight_decay=weight_decay, log_sampling=log_sampling,
                          polar_fourier=polar_fourier, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = SDFLoss(code_reg_weight=code_reg_weight, sum=True)
    id_range = torch.arange(0, max_id + 2, dtype=torch.long).to(device)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        print(torch.std(model.poly_embedding_layer.weight).item())
        model.train()
        epoch_loss = 0
        for id, sample, dist in tqdm(dataloader):
            id = id.to(device)
            sample = sample.to(device)
            dist = dist.to(device)
            optimizer.zero_grad()
            output = model(id, sample)
            latend_code = model.poly_embedding_layer(id_range)
            loss = loss_fn(output, dist, latend_code)  # , latend_code
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        test_epoch_loss = 0
        with torch.no_grad():
            model.eval()
            for id, sample, dist in val_dataloader:
                id = id.to(device)
                sample = sample.to(device)
                dist = dist.to(device)
                output = model(id, sample)
                loss = F.l1_loss(output, dist, reduction='mean')
                test_epoch_loss += loss.item()

        if best_val_loss > test_epoch_loss:
            torch.save(model.state_dict(), save_file_name)
            location_embedding = model.poly_embedding_layer.weight.data.cpu().numpy()
            np.save(save_file_name.replace('.pth', '_loc'), location_embedding)
            best_val_loss = test_epoch_loss

            if args.test_representation_location:
                test_representation.test_distance(polys_dict_loc, location_embedding, num_training=1, num_epochs=30,
                                                  num_pairs=50000)
            if args.visualSDF_location:
                visualization.random_visualization(polys_dict_loc, model)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}, TEST Loss: {test_epoch_loss}')

    z_size = args.z_size_shape
    hidden_size = args.hidden_size_shape
    num_freqs = args.num_freqs_shape
    code_reg_weight = args.code_reg_weight_shape
    weight_decay = args.weight_decay_shape
    num_layers = args.num_layers_shape
    epochs = args.epochs_shape
    polar_fourier = args.polar_fourier_shape
    log_sampling = args.log_sampling_shape

    training_ratio = args.training_ratio_shape
    samples_perUnit = args.samples_perUnit_shape
    point_sample = args.point_sample_shape
    sample_band_width = args.sample_band_width_shape
    uniformed_sample_perUnit = args.uniformed_sample_perUnit_shape

    multiprocessing.set_start_method("spawn", force=True)
    samples = MP_Sampling.MP_sample(polys_dict_shape, num_process, samples_perUnit=samples_perUnit,
                                    point_sample=point_sample,
                                    sample_band_width=sample_band_width,
                                    uniformed_sample_perUnit=uniformed_sample_perUnit)

    max_id = max(polys_dict_shape.keys())
    total_dataset = Geo2Vec_Dataset(samples, polys_dict_shape.keys())
    samples = None
    gc.collect()

    train_size = round(training_ratio * len(total_dataset))
    val_size = len(total_dataset) - train_size
    train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])
    total_dataset = None
    gc.collect()
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                            num_workers=num_workers,
                            pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                num_workers=num_workers,
                                pin_memory=True)

    print(f"In average training samples per entity: {len(train_dataset) / len(polys_dict_shape)}")

    model = Geo2Vec_Model(n_poly=max_id + 2, z_size=z_size, hidden_size=hidden_size, num_freqs=num_freqs,
                          weight_decay=weight_decay, log_sampling=log_sampling,
                          polar_fourier=polar_fourier, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = SDFLoss(code_reg_weight=code_reg_weight, sum=True)
    id_range = torch.arange(0, max_id + 2, dtype=torch.long).to(device)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        print(torch.std(model.poly_embedding_layer.weight).item())
        model.train()
        epoch_loss = 0
        for id, sample, dist in tqdm(dataloader):
            id = id.to(device)
            sample = sample.to(device)
            dist = dist.to(device)
            optimizer.zero_grad()
            output = model(id, sample)
            latend_code = model.poly_embedding_layer(id_range)
            loss = loss_fn(output, dist, latend_code)  # , latend_code
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        test_epoch_loss = 0
        with torch.no_grad():
            model.eval()
            for id, sample, dist in val_dataloader:
                id = id.to(device)
                sample = sample.to(device)
                dist = dist.to(device)
                output = model(id, sample)
                loss = F.l1_loss(output, dist, reduction='mean')
                test_epoch_loss += loss.item()

        if best_val_loss > test_epoch_loss:
            torch.save(model.state_dict(), save_file_name)
            shape_embedding = model.poly_embedding_layer.weight.data.cpu().numpy()
            np.save(save_file_name.replace('.pth', '_shp'), shape_embedding)
            best_val_loss = test_epoch_loss

            if args.test_representation_shape:
                labels = num_edges_labels
                classification = True if type(labels[0]) == str else False
                poly_ids = list(labels.keys())
                random.shuffle(poly_ids)
                poly_ids_train = poly_ids[:int(0.7 * len(poly_ids))]
                poly_ids_test = poly_ids[int(0.7 * len(poly_ids)):int(0.85 * len(poly_ids))]
                poly_ids_val = poly_ids[int(0.85 * len(poly_ids)):]
                result_class = test_representation.test_representation_embed(shape_embedding, z_size, labels, device,
                                                                             poly_ids_train, poly_ids_test,
                                                                             poly_ids_val,
                                                                             batch_size=32, epochs=100,
                                                                             print_result=False,
                                                                             classification=classification)

            if args.visualSDF_shape:
                visualization.random_visualization(polys_dict_shape, model)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}, TEST Loss: {test_epoch_loss}')

    entity_embedding = np.concatenate((location_embedding, shape_embedding), axis=-1)
    np.save(save_file_name.replace('.pth', '_conbine'), entity_embedding)


if __name__ == '__main__':
    main()
