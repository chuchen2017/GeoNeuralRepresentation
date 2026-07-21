import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
import numpy as np
# from sys import path
# path.append('..')
import utils.test_representation as test_representation
from utils.config import load_config
from utils.data_loader import load_data
from utils.train_utils import sample_geo2vec_dataset, train_geo2vec_model
import utils.visualization as visualization

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'learn_location_rep.yaml')

def get_args():
    parser = argparse.ArgumentParser(description="Geo2vec Training Config")

    #file_path: where the data is stored in pkl or gpkg format
    file_path = r'data\ShapeClassification.gpkg'
    parser.add_argument('--file_path', type=str, default=file_path)
    #Save file path
    save_path = os.path.splitext(file_path)[0] + '.pth'
    parser.add_argument('--save_file_name', type=str, default=save_path)

    #Sampling Parameters
    parser.add_argument('--num_process', type=int, default=12)
    parser.add_argument('--samples_perUnit', type=int, default=4000)
    parser.add_argument('--point_sample', type=int, default=10)
    parser.add_argument('--sample_band_width', type=float, default=0.1)
    parser.add_argument('--uniformed_sample_perUnit', type=int, default=30)

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1024 * 20)
    parser.add_argument('--num_workers', type=int, default=8)  # Training dataload number of works
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--z_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_freqs', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--code_reg_weight', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--polar_fourier', action='store_true', default=False)
    parser.add_argument('--log_sampling', action='store_true', default=False)
    parser.add_argument('--training_ratio', type=float, default=0.95)

    # Testing options
    parser.add_argument('--test_representation', type=bool, default=False)
    parser.add_argument('--visualSDF', type=bool, default=True)
    return load_config(parser, default_config=DEFAULT_CONFIG)


def main():
    args = get_args()
    device = torch.device(args.device)
    save_file_name = args.save_file_name

    polys_dict_shape,polys_dict_loc, classification_labels, areas_labels, perimeters_labels, num_edges_labels = load_data(args.file_path)

    train_dataset, val_dataset, max_id = sample_geo2vec_dataset(
        polys_dict_loc,
        num_process=args.num_process,
        samples_perUnit=args.samples_perUnit,
        point_sample=args.point_sample,
        sample_band_width=args.sample_band_width,
        uniformed_sample_perUnit=args.uniformed_sample_perUnit,
        training_ratio=args.training_ratio,
    )
    model, location_embedding, _ = train_geo2vec_model(
        train_dataset, val_dataset, max_id, device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        z_size=args.z_size,
        hidden_size=args.hidden_size,
        num_freqs=args.num_freqs,
        num_layers=args.num_layers,
        code_reg_weight=args.code_reg_weight,
        weight_decay=args.weight_decay,
        polar_fourier=args.polar_fourier,
        log_sampling=args.log_sampling,
        save_model_path=save_file_name,
    )
    np.save(save_file_name.replace('.pth', '_loc'), location_embedding)

    if args.test_representation:
        test_representation.test_distance(polys_dict_loc, location_embedding, num_training=1, num_epochs=30, num_pairs=50000)
    if args.visualSDF:
        visualization.random_visualization(polys_dict_loc, model)


if __name__ == '__main__':
    main()
