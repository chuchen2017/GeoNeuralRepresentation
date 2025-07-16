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
from models.Geo2Vec import Geo2Vec_Model,Geo2Vec_Dataset,SDFLoss
from models import MP_Sampling
from utils.data_loader import load_data
import utils.visualization as visualization

def get_args():
    parser = argparse.ArgumentParser(description="Geo2vec Training Config")

    #file_path: where the data is stored in pkl or gpkg format
    file_path = r'D:\Research\ServerBackup\Universal\shp\USC_lines.shp'
    file_path = r'D:\Research\2025\codes\NeuralRepresentation\data\Singapore_total_data.gpkg'
    parser.add_argument('--file_path', type=str, default=file_path)
    #Save file path
    save_path = file_path.replace('.shp','.pth')
    parser.add_argument('--save_file_name', type=str, default=save_path)

    #Sampling Parameters
    parser.add_argument('--num_process', type=int, default=16)
    parser.add_argument('--samples_perUnit', type=int, default=100) #200
    parser.add_argument('--point_sample', type=int, default=20) #50
    parser.add_argument('--sample_band_width', type=float, default=0.1)
    parser.add_argument('--uniformed_sample_perUnit', type=int, default=10) #20

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1024 * 20)
    parser.add_argument('--num_workers', type=int, default=8)  # Training dataload number of works
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--z_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_freqs', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--code_reg_weight', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--polar_fourier', action='store_true', default=False)
    parser.add_argument('--log_sampling', action='store_true', default=True)
    parser.add_argument('--training_ratio', type=float, default=0.95)

    # Testing options
    parser.add_argument('--test_representation', type=bool, default=True)
    parser.add_argument('--visualSDF', type=bool, default=True)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)

    file_path = args.file_path
    save_file_name = args.save_file_name

    z_size = args.z_size
    hidden_size = args.hidden_size
    num_freqs = args.num_freqs
    code_reg_weight = args.code_reg_weight
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    epochs = args.epochs
    polar_fourier = args.polar_fourier
    log_sampling = args.log_sampling

    batch_size = args.batch_size
    training_ratio = args.training_ratio
    num_process = args.num_process
    num_workers = args.num_workers
    samples_perUnit = args.samples_perUnit
    point_sample = args.point_sample
    sample_band_width = args.sample_band_width
    uniformed_sample_perUnit = args.uniformed_sample_perUnit

    polys_dict_shape,polys_dict_loc, classification_labels, areas_labels, perimeters_labels, num_edges_labels = load_data(file_path)
    multiprocessing.set_start_method("spawn", force=True)
    samples = MP_Sampling.MP_sample(polys_dict_shape, num_process, samples_perUnit=samples_perUnit, point_sample=point_sample,
                                    sample_band_width=sample_band_width,uniformed_sample_perUnit=uniformed_sample_perUnit)


    max_id = max(polys_dict_shape.keys())
    total_dataset = Geo2Vec_Dataset(samples, polys_dict_shape.keys())
    samples = None
    gc.collect()

    train_size = round(training_ratio * len(total_dataset))
    val_size = len(total_dataset) - train_size
    train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])
    total_dataset = None
    gc.collect()
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers,
                            pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                                pin_memory=True)

    print(f"In average training samples per entity: {len(train_dataset) / len(polys_dict_shape)}")

    model = Geo2Vec_Model(n_poly=max_id + 2, z_size=z_size, hidden_size=hidden_size, num_freqs=num_freqs,
                                      weight_decay=weight_decay, log_sampling=log_sampling,
                                      polar_fourier=polar_fourier, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = SDFLoss(code_reg_weight=code_reg_weight,sum=True)
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
            np.save(save_file_name.replace('.pth','_shp'), location_embedding)
            best_val_loss = test_epoch_loss

            if args.test_representation:
                labels = num_edges_labels
                classification = True if type(labels[0]) == str else False
                poly_ids = list(labels.keys())
                random.shuffle(poly_ids)
                poly_ids_train = poly_ids[:int(0.7 * len(poly_ids))]
                poly_ids_test = poly_ids[int(0.7 * len(poly_ids)):int(0.85 * len(poly_ids))]
                poly_ids_val = poly_ids[int(0.85 * len(poly_ids)):]
                result_class = test_representation.test_representation_embed(location_embedding, z_size, labels, device,
                                                                             poly_ids_train, poly_ids_test,
                                                                             poly_ids_val,
                                                                             batch_size=32, epochs=100,
                                                                             print_result=False,
                                                                             classification=classification)

            if args.visualSDF:
                visualization.random_visualization(polys_dict_shape,model)



        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}, TEST Loss: {test_epoch_loss }')


if __name__ == '__main__':
    main()
