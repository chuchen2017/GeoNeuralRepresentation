import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from models.sample_function import signed_distance

def plot_polygon(poly, linewidth=1):
    if poly.geom_type == 'Polygon':
        x, y = poly.exterior.xy
        plt.plot(x, y, color='black', linewidth=linewidth)
        for interior in poly.interiors:
            x, y = interior.xy
            plt.plot(x, y, color='yellow', linewidth=linewidth)
    elif poly.geom_type == 'MultiPolygon':
        for p in poly.geoms:
            x, y = p.exterior.xy
            plt.plot(x, y, color='black', linewidth=linewidth)
            for interior in p.interiors:
                x, y = interior.xy
                plt.plot(x, y, color='yellow', linewidth=linewidth)
    elif poly.geom_type == 'LineString':
        x, y = poly.xy
        plt.plot(x, y, color='black', linewidth=linewidth)
    elif poly.geom_type == 'MultiLineString':
        for p in poly.geoms:
            x, y = p.exterior.xy
            plt.plot(x, y, color='black', linewidth=linewidth)
    elif poly.geom_type == 'Point':
        x, y = poly.coords.xy
        plt.scatter(x, y, color='black', s=linewidth * 10)

def visualize_signed_distance_real(poly, bounds=((-1, 1), (-1, 1)), resolution=300, levels=100, countour=True):
    x_grid = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y_grid = np.linspace(bounds[1][0], bounds[1][1], resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    distances = []
    for i in range(len(grid_points)):
        distance = signed_distance(pt=grid_points[i], polygon=poly)
        distances.append(distance)
    distances = np.array(distances).reshape(xx.shape)

    if countour:
        plt.figure(figsize=(8, 7))
        contour = plt.contourf(xx, yy, distances, levels=levels, cmap='Spectral', alpha=0.9)
        plt.colorbar(contour, label='Signed Distance')
    else:
        plt.figure(figsize=(8, 7))
        plt.imshow(distances, extent=(bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]),
                   origin='lower', cmap='Spectral', alpha=0.9, aspect='auto')
        plt.colorbar(label='Signed Distance')

    plt.xticks([])
    plt.yticks([])
    plt.title('Signed Distance Field')
    plt.xlim(bounds[0][0], bounds[0][1])
    plt.ylim(bounds[1][0], bounds[1][1])
    plt.show()

def visualize_signed_distance(model, poly_id,polygon, bounds=((-5, 5), (-5, 5)), resolution=300,device=torch.device('cuda')):
    x_grid = torch.linspace(bounds[0][0], bounds[0][1], resolution)
    y_grid = torch.linspace(bounds[1][0], bounds[1][1], resolution)
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')  # match NumPy's default 'xy' behavior
    grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(device)

    model.eval()
    id_tensor = torch.tensor([poly_id] * grid_points.shape[0], dtype=torch.long).to(device)
    with torch.no_grad():
        preds = model(id_tensor, grid_points).cpu().numpy()
    preds = preds.reshape(xx.shape)
    plt.figure(figsize=(8, 7))
    contour = plt.contourf(xx, yy, preds, levels=100, cmap='Spectral', alpha=0.9)
    plt.colorbar(contour, label='Signed Distance')
    #plt.title("MLP Prediction of Signed Distance to Polygon with Hole")
    #plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.xlim(bounds[0][0], bounds[0][1])
    plt.ylim(bounds[1][0], bounds[1][1])

    plot_polygon(polygon, linewidth=1)
    #plt.show()

def visualize_signed_distance_w_embedding(model,embedding,device=torch.device('cuda'), bounds=((-5, 5), (-5, 5)), resolution=300):
    x_grid = torch.linspace(bounds[0][0], bounds[0][1], resolution)
    y_grid = torch.linspace(bounds[1][0], bounds[1][1], resolution)
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')  # match NumPy's default 'xy' behavior
    grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(device)

    model.eval()
    if embedding.shape[0] == 1:
        embedding = embedding.repeat(grid_points.shape[0], 1)
    with torch.no_grad():
        preds = model(embedding, grid_points).cpu().numpy()
    preds = preds.reshape(xx.shape)

    plt.figure(figsize=(8, 7))
    contour = plt.contourf(xx, yy, preds, levels=100, cmap='Spectral', alpha=0.9)
    plt.colorbar(contour, label='Signed Distance')

    plt.title("MLP Prediction of Signed Distance to Polygon with Hole")
    #plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.xlim(bounds[0][0], bounds[0][1])
    plt.ylim(bounds[1][0], bounds[1][1])
    #plt.show()

def random_visualization(polys_dict,model):
    poly_id = random.choice(list(polys_dict.keys()))
    poly=polys_dict[poly_id]
    delta = 0.2
    if poly.geom_type == 'Polygon':
        xs = [x[0] for x in poly.exterior.coords]
        ys = [x[1] for x in poly.exterior.coords]
    elif poly.geom_type == 'MultiPolygon':
        # Use the largest polygon in the MultiPolygon
        largest = max(poly.geoms, key=lambda p: p.area)
        xs = [x[0] for x in largest.exterior.coords]
        ys = [x[1] for x in largest.exterior.coords]
    else:
        # Fallback to centroid for other geometries like Point, LineString, etc.
        centroid = poly.centroid
        xs = [centroid.x]
        ys = [centroid.y]
    # while len(xs) < 15:
    #     poly_id = random.choice(list(polys_dict.keys()))
    #     xs = [x[0] for x in polys_dict[poly_id].exterior.coords]
    #     ys = [x[1] for x in polys_dict[poly_id].exterior.coords]
    bounds = ((min(xs) - delta, max(xs) + delta), (min(ys) - delta, max(ys) + delta))
    visualize_signed_distance(model, poly_id, polygon = poly, bounds=bounds, resolution=300)
    plt.show()
    #plt.savefig('../pics/' + poly_id + '.png',)