from utils.preprocess import count_edges
from models.sample_function import sample_signed_distance,sample_bounding_distance
from tqdm import tqdm
import multiprocessing
from functools import partial

def sample(poly_input,args):
    poly_id, poly = poly_input
    bounding_box, samples_perUnit, point_sample, sample_band_width, uniformed_sample_perUnit = args
    num_edges = count_edges(poly)
    if num_edges > float('inf'): #could set to sample more when greater than 200
        samples_perUnit = samples_perUnit*2
        point_sample = point_sample*2  # point_sample_global
        sample_band_width = sample_band_width
    else:
        samples_perUnit = samples_perUnit
        point_sample = point_sample
        sample_band_width = sample_band_width
    samples, distance = sample_signed_distance(poly, samples_perUnit=samples_perUnit, point_sample=point_sample,
                                               sample_band_width=sample_band_width, bounding_box=bounding_box)
    return (poly_id, samples, distance)

def sample_bounding(poly_input,args):
    poly_id, poly = poly_input
    bounding_box, samples_perUnit, point_sample, sample_band_width, uniformed_sample_perUnit = args
    num_edges = count_edges(poly)
    if num_edges > float('inf'): #could set to sample more when greater than 200
        extra_sample_perUnit = uniformed_sample_perUnit*2
    else:
        extra_sample_perUnit = uniformed_sample_perUnit
    samples, distance = sample_bounding_distance(poly, bounding_box, samples_perUnit=extra_sample_perUnit)
    return (poly_id, samples, distance)

def MP_sample(polys_dict,num_process,samples_perUnit=50,point_sample=50,sample_band_width=0.1,uniformed_sample_perUnit=20, buffer=0.1):
    minx = min(poly.bounds[0] for poly in polys_dict.values())
    maxx = max(poly.bounds[2] for poly in polys_dict.values())
    miny = min(poly.bounds[1] for poly in polys_dict.values())
    maxy = max(poly.bounds[3] for poly in polys_dict.values())
    bounding_box = (minx, maxx, miny, maxy)
    bounding_box = (bounding_box[0] - buffer, bounding_box[1] + buffer, bounding_box[2] - buffer, bounding_box[3] + buffer)
    print(f"Bounding box: {bounding_box}")

    args = (bounding_box,samples_perUnit,point_sample,sample_band_width,uniformed_sample_perUnit)

    partial_sample = partial(sample,args=args)
    partial_sample_bounding = partial(sample_bounding,args=args)

    total_samples = {}
    if len(polys_dict) > 10000:
        division = 5
    else:
        division = 2
    for i in range(division):
        print(f"Sampling progress: {i/division*100:.1f}%")
        polys_dict_subset = {k: v for k, v in polys_dict.items() if hash(k) % division == i}

        with multiprocessing.Pool(num_process) as pool:
            results = list(tqdm(pool.imap_unordered(partial_sample, polys_dict_subset.items()), total=len(polys_dict_subset)))

        training_samples = {}
        for poly_id, samples, distance in results:
            training_samples[poly_id] = (samples, distance)


        with multiprocessing.Pool(num_process) as pool:
            results = list(tqdm(pool.imap_unordered(partial_sample_bounding, polys_dict_subset.items()), total=len(polys_dict_subset)))

        for poly_id, samples, distance in results:
            samples1,distance1 = training_samples[poly_id]
            total_samples[poly_id] = (samples1+samples, distance1+distance)

    return total_samples
