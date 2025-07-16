import geopandas as gpd
import pickle
from tqdm import tqdm
from utils.preprocess import poly_preprocess,plot_polygon,count_edges,normalize_geometries
import random
import pandas as pd

import geopandas as gpd
import pandas as pd
from shapely.geometry import base
import os

def ensure_geodataframe(polys_scaled_normalized):
    # Case 1: Already a GeoDataFrame
    if isinstance(polys_scaled_normalized, gpd.GeoDataFrame):
        if polys_scaled_normalized.geometry.name != 'geometry':
            polys_scaled_normalized = polys_scaled_normalized.set_geometry('geometry')
        return polys_scaled_normalized

    # Case 2: File path (Shapefile or GeoJSON)
    if isinstance(polys_scaled_normalized, str) and os.path.isfile(polys_scaled_normalized):
        ext = os.path.splitext(polys_scaled_normalized)[1].lower()
        if ext in ['.shp', '.json', '.geojson']:
            gdf = gpd.read_file(polys_scaled_normalized)
            if gdf.geometry.name != 'geometry':
                gdf = gdf.set_geometry('geometry')
            return gdf
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Only .shp, .json, and .geojson are supported.")

    # Case 3: Pandas DataFrame with a 'geometry' column
    if isinstance(polys_scaled_normalized, pd.DataFrame):
        if 'geometry' in polys_scaled_normalized.columns:
            return gpd.GeoDataFrame(polys_scaled_normalized, geometry='geometry')

    # Case 4: List or Series of Shapely geometries
    if isinstance(polys_scaled_normalized, (list, pd.Series)) and all(isinstance(geom, base.BaseGeometry) for geom in polys_scaled_normalized):
        return gpd.GeoDataFrame(geometry=polys_scaled_normalized)

    raise TypeError("Cannot convert polys_scaled_normalized to GeoDataFrame. Input format not recognized.")


def load_data(dataset_name,visual=False):
    if dataset_name == 'Building':
        polys_scaled_normalized = gpd.read_file('../data/ShapeClassification.gpkg')
    elif dataset_name == 'MNIST':
        polys_scaled_normalized = gpd.read_file('../data/MNIST.gpkg')
    elif dataset_name == 'Singapore':
        # Load the polys_dict from the pickle file
        with open('../data/Singapore_total_data.pkl', 'rb') as f:
            polys_scaled_normalized = pickle.load(f)
        polys_scaled_normalized = [data['shape'] for data in polys_scaled_normalized]
    elif dataset_name == 'NYC':
        with open('/home/users/chen/2024/NeuralRepresentation/data/Polygon/NYC_total_data.pkl', 'rb') as f:
            polys_scaled_normalized = pickle.load(f)
        polys_scaled_normalized = [data['shape'] for data in polys_scaled_normalized]
    else:
        if '.pkl' in dataset_name:
            with open(dataset_name, 'rb') as f:
                polys_scaled_normalized = pickle.load(f)
            polys_scaled_normalized = [data['shape'] for data in polys_scaled_normalized]
        elif '.gpkg' in dataset_name:
            polys_scaled_normalized = gpd.read_file(dataset_name)
        else:
            raise ValueError(
                f"Unsupported Datatype, Define your dataloader, Load the data as gpd.")

    if type(polys_scaled_normalized) is list:
        # If polys_scaled_normalized is a list, convert it to a GeoDataFrame
        polys_scaled_normalized = gpd.GeoDataFrame(geometry=polys_scaled_normalized)
    polys_scaled_normalized = ensure_geodataframe(polys_scaled_normalized)

    minx, miny, maxx, maxy = polys_scaled_normalized.total_bounds
    # Check if any bound exceeds 1.0 in absolute value
    if any(coord > 1.1 for coord in [abs(minx), abs(miny), abs(maxx), abs(maxy)]):
        print("Normalizing geometries...")
        polys_scaled_normalized = normalize_geometries(list(polys_scaled_normalized['geometry'].values))
        polys_scaled_normalized = gpd.GeoDataFrame(geometry=polys_scaled_normalized)

    polys_dict_shape = {}
    polys_dict_location = {}
    classification_labels = {}
    areas_labels = {}
    perimeters_labels = {}
    num_edges_labels = {}
    for id, row in tqdm(polys_scaled_normalized.iterrows()):
        poly = row['geometry']
        preprocess = poly_preprocess(poly)
        polys_dict_shape[id] = preprocess[0]#[0]#[1]
        polys_dict_location[id] = preprocess[1]
        if 'label' in row:
            classification_labels[id] = str(row['label'])
        areas_labels[id] = polys_dict_location[id].area
        perimeters_labels[id] = polys_dict_location[id].length

        num_edges_labels[id] = count_edges(polys_dict_location[id])

        if visual and random.random() < 0.001:
            plot_polygon(polys_dict_location[id])

    return polys_dict_shape,polys_dict_location, classification_labels, areas_labels, perimeters_labels, num_edges_labels