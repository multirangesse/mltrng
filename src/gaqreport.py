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

#Generating the Gradual Anisotropic Queries
def random_gaquery_sample(max_dim):
    width = np.random.randint(1,int(max_dim/4)+1)
    c = np.random.randint(1,int((max_dim/4)/width)+1)
    rn1 = random_range_calc(c * width, max_dim)
    rn2 = random_range_calc(2 * c * width, max_dim)
    rn3 = random_range_calc(3 * c * width, max_dim)
    rn4 = random_range_calc(4 * c * width, max_dim)
    gaquery = np.random.permutation( (rn1, rn2, rn3, rn4) )
    return tuple(map(tuple, gaquery))

seed_every_thing(20)
sample_num = 1000
max_dim = 16
range_list = [random_gaquery_sample(max_dim) for i in range(sample_num)]
range_list = list(set(range_list))
while len(range_list)<sample_num:
    range_list = list(set(range_list))
    sample_diff = sample_num - len(range_list)
    for i in range(sample_diff):
        range_list.append( random_gaquery_sample(max_dim) )


class args:
    datagen = None
    encrypt = None
    kdisk = None
    shuffle = None
    report = True
    datadim = [4, 16, 16, 16, 16]
    slabdim = [4, 4, 4, 4]
    query = [None]
    rangequery = [0, 2, 0, 2, 0, 2, 0, 2]
    slab_size = 256

for single_range in tqdm(range_list):
    args.rangequery = list(np.array([list(dim_range) for dim_range in single_range]).flatten())
    report_hist(args, indx =  np.prod(args.datadim[1:]))