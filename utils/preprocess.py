from shapely.affinity import translate, scale
import numpy as np
import shapely.geometry
from shapely import affinity
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString, Point, MultiPoint, GeometryCollection
)
from shapely.geometry.base import BaseGeometry

def count_edges(geom: BaseGeometry) -> int:
    if geom.is_empty:
        return 0

    if isinstance(geom, LineString):
        return max(0, len(geom.coords) - 1)

    elif isinstance(geom, Polygon):
        count = max(0, len(geom.exterior.coords) - 1)
        count += sum(max(0, len(interior.coords) - 1) for interior in geom.interiors)
        return count

    elif isinstance(geom, MultiPolygon):
        return sum(count_edges(part) for part in geom.geoms)

    elif isinstance(geom, MultiLineString):
        return sum(count_edges(part) for part in geom.geoms)

    elif isinstance(geom, GeometryCollection):
        return sum(count_edges(part) for part in geom.geoms)

    # Points or MultiPoints have no edges
    elif isinstance(geom, (Point, MultiPoint)):
        return 0

    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")

def normalize_geometries(polys_list):
    """
    Normalize all the polygons in the list to (-1,1)
    Keep distance relationship invariant
    """
    # Step 1: Get total bounds
    all_bounds = [geom.bounds for geom in polys_list]
    min_x = min(b[0] for b in all_bounds)
    min_y = min(b[1] for b in all_bounds)
    max_x = max(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)

    width = max_x - min_x
    height = max_y - min_y

    # Avoid division by zero
    if width == 0:
        width = 1e-9
    if height == 0:
        height = 1e-9


    normalized_dict = []
    # Step 2: Normalize each geometry
    for geom in polys_list:
        # Translate to origin
        translated = translate(geom, xoff=-min_x, yoff=-min_y)
        # Scale to unit square
        normalized = scale(translated, xfact=1 / width, yfact=1 / height, origin=(0, 0))
        normalized_dict.append(normalized)
    return normalized_dict

def normalize_coords(coords, center_x, center_y, scale):
    return [((x - center_x) / scale, (y - center_y) / scale) for x, y in coords]

def poly_preprocess(poly):
    """
    Normalize a entity to -1,1
    """
    if poly.geom_type == 'Polygon':
        xs = poly.exterior.coords.xy[0]
        ys = poly.exterior.coords.xy[1]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        scale_x = max_x - min_x
        scale_y = max_y - min_y
        scale = max(scale_x, scale_y)
        if scale == 0:
            scale = 1

        exterior = normalize_coords(poly.exterior.coords, center_x, center_y, scale)
        interiors = [normalize_coords(interior.coords, center_x, center_y, scale) for interior in poly.interiors]

        new_poly = Polygon(exterior, interiors)
        return (new_poly, poly, scale, center_x, center_y)

    elif poly.geom_type == 'MultiPolygon':
        all_x = [x for p in poly.geoms for x in p.exterior.coords.xy[0]]
        all_y = [y for p in poly.geoms for y in p.exterior.coords.xy[1]]

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        scale_x = max_x - min_x
        scale_y = max_y - min_y
        scale = max(scale_x, scale_y)
        if scale == 0:
            scale = 1

        new_polys = []
        for p in poly.geoms:
            exterior = normalize_coords(p.exterior.coords, center_x, center_y, scale)
            interiors = [normalize_coords(interior.coords, center_x, center_y, scale) for interior in p.interiors]
            new_polys.append(Polygon(exterior, interiors))

        new_multi = MultiPolygon(new_polys)
        return (new_multi, poly, scale, center_x, center_y)

    elif poly.geom_type == 'LineString':
        xs = poly.coords.xy[0]
        ys = poly.coords.xy[1]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        scale_x = max_x - min_x
        scale_y = max_y - min_y
        scale = max(scale_x, scale_y)
        if scale == 0:
            scale = 1

        new_line = LineString(normalize_coords(poly.coords, center_x, center_y, scale))
        return (new_line, poly, scale, center_x, center_y)

    elif poly.geom_type == 'Point':
        x, y = poly.x, poly.y
        center_x = x
        center_y = y
        scale = 1  # no scaling needed

        new_point = Point(0.0, 0.0)  # centered at origin
        return (new_point, poly, scale, center_x, center_y)

    else:
        raise ValueError(f"Unsupported geometry type: {poly.geom_type}")

def rotation(polygon):
    rectangle = polygon.minimum_rotated_rectangle
    xc = polygon.centroid.x
    yc = polygon.centroid.y
    rec_x = []
    rec_y = []
    for point in rectangle.exterior.coords:
        rec_x.append(point[0])
        rec_y.append(point[1])
    top = np.argmax(rec_y)
    top_left = top - 1 if top > 0 else 3
    top_right = top + 1 if top < 3 else 0
    x0, y0 = rec_x[top], rec_y[top]
    x1, y1 = rec_x[top_left], rec_y[top_left]
    x2, y2 = rec_x[top_right], rec_y[top_right]
    d1 = np.linalg.norm([x0 - x1, y0 - y1])
    d2 = np.linalg.norm([x0 - x2, y0 - y2])
    if d1 > d2:
        cosp = (x1 - x0) / d1
        sinp = (y0 - y1) / d1
    else:
        cosp = (x2 - x0) / d2
        sinp = (y0 - y2) / d2
    # rotations[i] = [cosp, sinp]
    matrix = (cosp, -sinp, 0.0,
              sinp, cosp, 0.0,
              0.0, 0.0, 1.0,
              xc - xc * cosp + yc * sinp, yc - xc * sinp - yc * cosp, 0.0)
    polygon = affinity.affine_transform(polygon, matrix)
    return polygon, [cosp, sinp]