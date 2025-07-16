import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def visualize_signed_distance(model, poly_id,polygon, bounds=((-5, 5), (-5, 5)), resolution=300,device=torch.device('cuda')):
    x_grid = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y_grid = np.linspace(bounds[1][0], bounds[1][1], resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    model.eval()
    id_tensor = torch.tensor([poly_id] * grid_points.shape[0], dtype=torch.long).to(device)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(id_tensor, grid_points_tensor).cpu().numpy()
    preds = preds.reshape(xx.shape)

    plt.figure(figsize=(8, 7))
    contour = plt.contourf(xx, yy, preds, levels=100, cmap='Spectral', alpha=0.9)
    plt.colorbar(contour, label='Signed Distance')

    if polygon.geom_type == 'Polygon':
        x_poly, y_poly = polygon.exterior.xy
        plt.plot(x_poly, y_poly, 'k-', lw=0.2)
        for interior in polygon.interiors:
            x_int, y_int = interior.xy
            plt.plot(x_int, y_int, 'k--', lw=0.2)
    elif polygon.geom_type == 'LineString':
        x_poly, y_poly = polygon.xy
        plt.plot(x_poly, y_poly, 'k-', lw=0.2)
    elif polygon.geom_type == 'Point':
        x_poly, y_poly = polygon.xy
        plt.plot(x_poly, y_poly, 'ro', markersize=5)

    plt.title("MLP Prediction of Signed Distance to Polygon with Hole")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.xlim(bounds[0][0], bounds[0][1])
    plt.ylim(bounds[1][0], bounds[1][1])
    #plt.show()

def visualize_signed_distance_w_embedding(model,embedding,device=torch.device('cuda'), bounds=((-5, 5), (-5, 5)), resolution=300):
    x_grid = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y_grid = np.linspace(bounds[1][0], bounds[1][1], resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    model.eval()
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)

    if embedding.shape[0] == 1:
        embedding = embedding.repeat(grid_points_tensor.shape[0], 1)
    with torch.no_grad():
        preds = model(embedding, grid_points_tensor).cpu().numpy()
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
    xs = [x[0] for x in poly.exterior.coords] if poly.geom_type == 'Polygon' else poly.coords
    ys = [x[1] for x in poly.exterior.coords] if poly.geom_type == 'Polygon' else poly.coords
    # while len(xs) < 15:
    #     poly_id = random.choice(list(polys_dict.keys()))
    #     xs = [x[0] for x in polys_dict[poly_id].exterior.coords]
    #     ys = [x[1] for x in polys_dict[poly_id].exterior.coords]
    bounds = ((min(xs) - delta, max(xs) + delta), (min(ys) - delta, max(ys) + delta))
    visualize_signed_distance(model, poly_id, polygon = polys_dict[poly_id], bounds=bounds, resolution=300)
    plt.show()