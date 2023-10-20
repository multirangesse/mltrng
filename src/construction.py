import numpy as np

from utils.report import report

class args:
    datagen = None
    encrypt = None
    kdisk = None
    shuffle = None
    report = True
    datadim = [4, 4, 4, 4, 4]
    slabdim = [2, 2, 2, 2]
    query = [None]
    rangequery = [0, 2, 0, 2, 0, 2, 0, 2]
    slab_size = 16


report(args, prompt = True, indx =  np.prod(args.datadim[1:]))