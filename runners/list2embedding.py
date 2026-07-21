import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import argparse
import torch
# from sys import path
# path.append('..')
from utils.config import load_config
from utils.data_loader import preprocessing_list
from utils.train_utils import sample_geo2vec_dataset, train_geo2vec_model

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'list2embedding.yaml')


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
    parser.add_argument('--visualSDF_location', type=bool, default=False)
    # For shape
    parser.add_argument('--test_representation_shape', type=bool, default=True)
    parser.add_argument('--visualSDF_shape', type=bool, default=False)
    return load_config(parser, default_config=DEFAULT_CONFIG)


def list2vec(Geolist,save_model_path=None,Geo_dim=128,num_epoch = None,location_learning=True,shape_learning=True,save_file_name=None,args=None):
    args = get_args() if args is None else args
    device = torch.device(args.device)

    torch.set_num_threads(args.num_process)

    polys_dict_shape, polys_dict_loc, classification_labels, areas_labels, perimeters_labels, num_edges_labels = preprocessing_list(Geolist)

    if location_learning:
        train_dataset, val_dataset, max_id = sample_geo2vec_dataset(
            polys_dict_loc,
            num_process=args.num_process,
            samples_perUnit=args.samples_perUnit_location,
            point_sample=args.point_sample_location,
            sample_band_width=args.sample_band_width_location,
            uniformed_sample_perUnit=args.uniformed_sample_perUnit_location,
            training_ratio=args.training_ratio_location,
        )
        _, location_embedding, _ = train_geo2vec_model(
            train_dataset, val_dataset, max_id, device,
            epochs=num_epoch if num_epoch is not None else args.epochs_location,
            batch_size=args.batch_size,
            z_size=Geo_dim if Geo_dim is not None else args.z_size_location,
            hidden_size=args.hidden_size_location,
            num_freqs=args.num_freqs_location,
            num_layers=args.num_layers_location,
            code_reg_weight=args.code_reg_weight_location,
            weight_decay=args.weight_decay_location,
            polar_fourier=args.polar_fourier_location,
            log_sampling=args.log_sampling_location,
            save_model_path=save_model_path.replace('.pth', '_loc.pth') if save_model_path is not None else None,
        )

    if shape_learning:
        train_dataset, val_dataset, max_id = sample_geo2vec_dataset(
            polys_dict_shape,
            num_process=args.num_process,
            samples_perUnit=args.samples_perUnit_shape,
            point_sample=args.point_sample_shape,
            sample_band_width=args.sample_band_width_shape,
            uniformed_sample_perUnit=args.uniformed_sample_perUnit_shape,
            training_ratio=args.training_ratio_shape,
        )
        _, shape_embedding, _ = train_geo2vec_model(
            train_dataset, val_dataset, max_id, device,
            epochs=num_epoch if num_epoch is not None else args.epochs_shape,
            batch_size=args.batch_size,
            z_size=Geo_dim if Geo_dim is not None else args.z_size_shape,
            hidden_size=args.hidden_size_shape,
            num_freqs=args.num_freqs_shape,
            num_layers=args.num_layers_shape,
            code_reg_weight=args.code_reg_weight_shape,
            weight_decay=args.weight_decay_shape,
            polar_fourier=args.polar_fourier_shape,
            log_sampling=args.log_sampling_shape,
            save_model_path=save_model_path.replace('.pth', '_shp.pth') if save_model_path is not None else None,
        )

    if location_learning and shape_learning:
        entity_embedding = np.concatenate((location_embedding, shape_embedding), axis=-1)
    elif shape_learning:
        entity_embedding = shape_embedding
    elif location_learning:
        entity_embedding = location_embedding
    else:
        entity_embedding = np.zeros((len(Geolist),Geo_dim))

    if save_file_name is not None:
        np.save(save_file_name, entity_embedding)
    else:
        pass
    return entity_embedding
