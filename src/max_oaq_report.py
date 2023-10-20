import random

import numpy as np

from utils.queryreport import report_hist
from tqdm import tqdm


def seed_every_thing(seed_num):
    np.random.seed(seed_num)
    random.seed(seed_num)
    
def random_range_calc(width, max_dim):
    start = np.random.randint(max_dim+1-width)
    return (start, start + width)

def random_oaquery_d1max_sample(max_dim,d_range = 5):
    width_min = np.random.randint(1,d_range+1)
    width_max = np.random.randint(max_dim-d_range,max_dim)
    rn1 = random_range_calc(width_max, max_dim)
    rn2 = random_range_calc(width_min, max_dim)
    rn3 = random_range_calc(width_min, max_dim)
    rn4 = random_range_calc(width_min, max_dim)
    return (rn1, rn2, rn3, rn4)

def random_oaquery_d2max_sample(max_dim,d_range = 5):
    width_min = np.random.randint(1,d_range+1)
    width_max = np.random.randint(max_dim-d_range,max_dim)
    rn1 = random_range_calc(width_min, max_dim)
    rn2 = random_range_calc(width_max, max_dim)
    rn3 = random_range_calc(width_min, max_dim)
    rn4 = random_range_calc(width_min, max_dim)
    return (rn1, rn2, rn3, rn4)

def random_oaquery_d3max_sample(max_dim,d_range = 5):
    width_min = np.random.randint(1,d_range+1)
    width_max = np.random.randint(max_dim-d_range,max_dim)
    rn1 = random_range_calc(width_min, max_dim)
    rn2 = random_range_calc(width_min, max_dim)
    rn3 = random_range_calc(width_max, max_dim)
    rn4 = random_range_calc(width_min, max_dim)
    return (rn1, rn2, rn3, rn4)

def random_oaquery_d4max_sample(max_dim,d_range = 5):
    width_min = np.random.randint(1,d_range+1)
    width_max = np.random.randint(max_dim-d_range,max_dim)
    rn1 = random_range_calc(width_min, max_dim)
    rn2 = random_range_calc(width_min, max_dim)
    rn3 = random_range_calc(width_min, max_dim)
    rn4 = random_range_calc(width_max, max_dim)
    return (rn1, rn2, rn3, rn4)

seed_every_thing(20)
sample_num = 100
max_dim = 32
range_list = [random_oaquery_d1max_sample(max_dim) for i in range(sample_num)]
range_list = list(set(range_list))
while len(range_list)<sample_num:
    range_list = list(set(range_list))
    sample_diff = sample_num - len(range_list)
    for i in range(sample_diff):
        range_list.append( random_oaquery_d1max_sample(max_dim) )


class args:
    datagen = None
    encrypt = None
    kdisk = None
    shuffle = None
    report = True
    datadim = [4, 32, 32, 32, 32]
    slabdim = [8, 8, 8, 8]
    query = [None]
    rangequery = [0, 2, 0, 2, 0, 2, 0, 2]
    slab_size = 4096

for single_range in tqdm(range_list):
    args.rangequery = list(np.array([list(dim_range) for dim_range in single_range]).flatten())
    report_hist(args, indx =  np.prod(args.datadim[1:]))