import numpy as np
from utils.data_loader import load_data
import random
import utils.test_representation as test_representation
import torch
from torch.utils.data import DataLoader, Dataset, random_split

def test_shp():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_path = '/home/users/chen/2024/NeuralRep/data/ShapeClassification.gpkg'
    polys_embedding = np.load('/home/users/chen/2024/NeuralRepresentation/data/Geo2vec/test.np.npy')
    num_training = 5

    polys_dict_shape, polys_dict_loc, classification_labels, areas_labels, perimeters_labels, num_edges_labels = load_data(file_path)
    labels = classification_labels  # num_edges_labels#classification_labels#classification_labels#num_edges_labels# classification_labels #

    classification = True if type(labels[0]) == str else False
    z_size = polys_embedding.shape[-1]
    acc_list = []
    for i in range(num_training):
        poly_ids = list(labels.keys())
        random.shuffle(poly_ids)
        poly_ids_train = poly_ids[:int(0.7 * len(poly_ids))]
        poly_ids_test = poly_ids[int(0.7 * len(poly_ids)):int(0.85 * len(poly_ids))]
        poly_ids_val = poly_ids[int(0.85 * len(poly_ids)):]
        result_class = test_representation.test_representation_embed(polys_embedding, z_size, labels, device,
                                                                     poly_ids_train, poly_ids_test, poly_ids_val,
                                                                     batch_size=32, epochs=100, print_result=False,
                                                                     classification=classification)
        if classification:
            initiation_accuracy_train, initiation_accuracy_test, initiation_accuracy_val, accuracy_train, accuracy_test, accuracy_val, false_predictions = result_class
            acc_list.append(accuracy_test)
        else:
            initiation_mse, initiation_mse_test, initiation_mse_val, t, mse_test, mse_val = result_class
            acc_list.append(mse_test)

    print('Average Result AND std', np.mean(acc_list), np.std(acc_list))



if __name__ == '__main__':
    test_shp()
