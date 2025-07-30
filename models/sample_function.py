import random
import numpy as np
import shapely.geometry
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.affinity import scale, translate


def signed_distance(pt: tuple, polygon: shapely.geometry) -> float:
    """
    Calculate the signed distance from a point to a polygon.
    Positive distance indicates the point is outside the polygon,
    negative distance indicates the point is inside the polygon.
    """
    point = shapely.geometry.Point(pt)  # Point(pt) # pt is a tuple (x, y)
    try:
        if polygon.geom_type == 'Polygon':
            distance = polygon.exterior.distance(point)
            for interior in polygon.interiors:
                distance = min(distance, interior.distance(point))
            return -distance if polygon.contains(point) else distance  # point.distance(polygon)
        elif polygon.geom_type == 'MultiPolygon':
            distance = float('inf')
            for poly in polygon.geoms:
                d = poly.exterior.distance(point)
                for interior in poly.interiors:
                    d = min(d, interior.distance(point))
                if poly.contains(point):
                    d = -d
                distance = min(distance, d)
            return distance
        elif polygon.geom_type == 'LineString':
            distance = polygon.distance(point)
            return distance  # point.distance(polygon)
        elif polygon.geom_type == 'MultiLineString':
            distance = float('inf')
            for poly in polygon.geoms:
                d = poly.distance(point)
                if d<distance:
                    distance = d
            return distance
        elif polygon.geom_type == 'Point':
            return point.distance(polygon)
    except Exception as e:
        print(f"Error calculating signed distance: {polygon}")
        return 0


def sample_perpendicular_at_fraction(x1, y1, x2, y2, f, d):
    # Line vector
    dx = x2 - x1
    dy = y2 - y1

    # Point at fraction f
    xt = x1 + f * dx
    yt = y1 + f * dy

    # Normalize perpendicular vector
    length_inv = 1.0 / np.hypot(dx, dy)
    perp_dx = -dy * length_inv
    perp_dy = dx * length_inv

    # Choose left/right randomly
    sign = np.random.choice([-1, 1])

    # Apply perpendicular offset
    x3 = xt + sign * d * perp_dx
    y3 = yt + sign * d * perp_dy

    return x3, y3


def sample_signed_distance(polygon, samples_perUnit=1000, point_sample=50, sample_band_width=0.5, bounding_box=None,
                           multi_polygon=None):
    if polygon.geom_type == 'Polygon':
        # sample all the points in the polygon
        poly_points = polygon.exterior.coords.xy
        poly_points = [(x, y) for x, y in zip(poly_points[0], poly_points[1])]
        signed_distances = [0] * len(poly_points)  # [signed_distance(pt, polygon) for pt in poly_points] # why not 0?

        sampled_points_boundary = []
        sampled_points_signed_dist = []

        for i in range(len(poly_points) - 1):
            p1 = poly_points[i]
            p2 = poly_points[i + 1]
            length = np.linalg.norm(np.array(p2) - np.array(p1))
            num_samples = int(length * samples_perUnit)

            for nn in range(point_sample):
                dx = random.gauss(0, sample_band_width)  # /2
                dy = random.gauss(0, sample_band_width)
                x3 = p1[0] + dx
                y3 = p1[1] + dy
                sampled_points_boundary.append((x3, y3))
                sampled_points_signed_dist.append(
                    signed_distance((x3, y3), polygon) if multi_polygon is None else signed_distance((x3, y3),
                                                                                                     multi_polygon))

            for sample in range(num_samples):
                f = random.uniform(0, 1)
                d = random.gauss(0, sample_band_width)
                x3, y3 = sample_perpendicular_at_fraction(p1[0], p1[1], p2[0], p2[1], f, d)
                sampled_points_boundary.append((x3, y3))
                sampled_points_signed_dist.append(
                    signed_distance((x3, y3), polygon) if multi_polygon is None else signed_distance((x3, y3),
                                                                                                     multi_polygon))

        # If the polygon has interiors, sample points in the interiors
        if polygon.interiors:
            for interior in polygon.interiors:
                int_points = interior.coords.xy
                int_points = [(x, y) for x, y in zip(int_points[0], int_points[1])]
                poly_points.extend(int_points)
                signed_distances.extend(
                    [0] * len(int_points))  # [signed_distance(pt, polygon) for pt in int_points] # why not 0?

                for i in range(len(int_points) - 1):
                    p1 = int_points[i]
                    p2 = int_points[i + 1]
                    length = np.linalg.norm(np.array(p2) - np.array(p1))
                    num_samples = int(length * samples_perUnit)

                    for nn in range(point_sample):
                        dx = random.gauss(0, sample_band_width)
                        dy = random.gauss(0, sample_band_width)
                        x3 = p1[0] + dx
                        y3 = p1[1] + dy
                        sampled_points_boundary.append((x3, y3))
                        sampled_points_signed_dist.append(
                            signed_distance((x3, y3), polygon) if multi_polygon is None else signed_distance((x3, y3),
                                                                                                             multi_polygon))
                    for sample in range(num_samples):
                        f = random.uniform(0, 1)
                        d = random.gauss(0, sample_band_width)
                        x3, y3 = sample_perpendicular_at_fraction(p1[0], p1[1], p2[0], p2[1], f, d)
                        sampled_points_boundary.append((x3, y3))
                        sampled_points_signed_dist.append(
                            signed_distance((x3, y3), polygon) if multi_polygon is None else signed_distance((x3, y3),
                                                                                                             multi_polygon))

        return poly_points + sampled_points_boundary, signed_distances + sampled_points_signed_dist

    elif polygon.geom_type == 'MultiPolygon':
        # sample all the points in the polygon
        poly_points = []
        signed_distances = []
        for poly in polygon.geoms:
            sample1, distance1 = sample_signed_distance(poly, samples_perUnit, point_sample, sample_band_width,
                                                        multi_polygon=polygon)
            poly_points.extend(sample1)
            signed_distances.extend(distance1)
        return poly_points, signed_distances

    elif polygon.geom_type == 'LineString':
        # sample all the points in the polygon
        poly_points = polygon.coords.xy
        poly_points = [(x, y) for x, y in zip(poly_points[0], poly_points[1])]
        signed_distances = [0] * len(poly_points)  # [signed_distance(pt, polygon) for pt in poly_points]

        sampled_points_boundary = []
        sampled_points_signed_dist = []

        for i in range(len(poly_points)):
            p1 = poly_points[i]
            for nn in range(point_sample):
                dx = random.gauss(0, sample_band_width)  # /2
                dy = random.gauss(0, sample_band_width)  # /2
                x3 = p1[0] + dx
                y3 = p1[1] + dy
                sampled_points_boundary.append((x3, y3))
                sampled_points_signed_dist.append(
                    signed_distance((x3, y3), polygon) if multi_polygon is None else signed_distance((x3, y3),
                                                                                                     multi_polygon))
        for i in range(len(poly_points) - 1):
            p1 = poly_points[i]
            p2 = poly_points[i + 1]
            length = np.linalg.norm(np.array(p2) - np.array(p1))
            num_samples = int(length * samples_perUnit)

            for sample in range(num_samples):
                f = random.uniform(0, 1)
                d = random.gauss(0, sample_band_width)
                x3, y3 = sample_perpendicular_at_fraction(p1[0], p1[1], p2[0], p2[1], f, d)
                sampled_points_boundary.append((x3, y3))
                sampled_points_signed_dist.append(
                    signed_distance((x3, y3), polygon) if multi_polygon is None else signed_distance((x3, y3),
                                                                                                     multi_polygon))

        return poly_points + sampled_points_boundary, signed_distances + sampled_points_signed_dist

    elif polygon.geom_type == 'MultiLineString':
        poly_points = []
        signed_distances = []
        for poly in polygon.geoms:
            sample1, distance1 = sample_signed_distance(poly, samples_perUnit, point_sample, sample_band_width,
                                                        multi_polygon=polygon)
            poly_points.extend(sample1)
            signed_distances.extend(distance1)
        return poly_points, signed_distances

    elif polygon.geom_type == 'Point':
        # sample all the points in the polygon
        poly_points = polygon.coords.xy
        poly_points = [(x, y) for x, y in zip(poly_points[0], poly_points[1])]
        signed_distances = [signed_distance(pt, polygon) for pt in poly_points]
        x1, y1 = poly_points[0]

        sampled_points_boundary = []
        sampled_points_signed_dist = []

        for nn in range(point_sample):
            dx = random.gauss(0, sample_band_width)  # /2
            dy = random.gauss(0, sample_band_width)  # /2
            x3 = x1 + dx
            y3 = y1 + dy
            sampled_points_boundary.append((x3, y3))
            sampled_points_signed_dist.append(signed_distance((x3, y3), polygon))

        return poly_points + sampled_points_boundary, signed_distances + sampled_points_signed_dist



    else:
        raise ValueError(
            f"Unsupported geometry type: {polygon.geom_type}. Supported types are Polygon, MultiPolygon, LineString, and Point.")

def sample_bounding_distance(polygon, bounds, samples_perUnit=1000):
    # bounds = (minx, maxx, miny, maxy)
    minx, maxx, miny, maxy = bounds
    x_grid = np.linspace(minx, maxx, samples_perUnit)
    y_grid = np.linspace(miny, maxy, samples_perUnit)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    samples = grid_points.tolist()
    signed_distances = [signed_distance((pt[0], pt[1]), polygon) for pt in samples]
    return samples, signed_distances

if __name__ == "__main__":
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], holes=[[(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)]])

    print("Signed distance from (0.5, 0.5) to polygon:", signed_distance((0.5, 0.5), poly))
    print("Signed distance from (1.5, 1.5) to polygon:", signed_distance((1.5, 1.5), poly))
