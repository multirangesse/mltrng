import random
import os
import logging
import linecache
import pickle

import numpy as np

from tqdm import tqdm
from crypto.encryption import AES_crypto, AES_XTS_crypto
from kdtree.kdtree import KdTree
from dataloader.dataloader import slab_matrix2numbers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)") 

class datagen:
    def __init__(self, 
                 dir_txt, 
                 data_dim=2, 
                 lat_dim=100, 
                 long_dim=100, 
                 height_dim=None, 
                 time_dim=None,
                 ) -> None:
        self.data_dim = data_dim
        self.lat_dim = lat_dim
        self.long_dim = long_dim
        self.height_dim = height_dim
        self.time_dim = time_dim
        self.fileNameTXT = dir_txt        
        self.seed = 42

        if self.data_dim==2:
            self.dataShape = (self.lat_dim, self.long_dim, self.data_dim+2)
        elif self.data_dim==3:
            self.dataShape = (self.lat_dim, self.long_dim, self.height_dim, self.data_dim+2)
        elif self.data_dim==4:
            self.dataShape = (self.lat_dim, self.long_dim, self.height_dim, self.time_dim, self.data_dim+2)

    def generate_data(self) -> None:  # generates Random Data        
        if self.data_dim == 2:
            self.__generate_data_2D()                
        if self.data_dim == 3:
            self.__generate_data_3D()                 
        if self.data_dim == 4:
            self.__generate_data_4D()  

        logging.info(f'The {self.fileNameTXT} generated!')
        return None
    
    def __generate_data_2D(self) -> None:  # Generates 2-Dimensional Random Dataset
        random.seed(self.seed)
        with open(self.fileNameTXT, "w") as file:
            file.write('lat' + '\t'*2+'long' +
                       '\t'+'temprature'+'\t'*4+'humidity'+'\n')
            for i in range(self.lat_dim):
                for j in range(self.long_dim):
                    temp = random.uniform(-20, 50)
                    humid = random.uniform(0, 100)
                    file.write(str(i)+'\t\t'+str(j) +
                               '\t\t'+str(temp)+'\t\t'+str(humid)+'\n')
        return None

    def __generate_data_3D(self) -> None:  # Generate 3-Dimensional Random Dataset
        random.seed(self.seed)
        with open(self.fileNameTXT, "w") as file:
            file.write('lat' + '\t'*2+'long' + '\t'+'height' +
                       '\t'+'temprature'+'\t'*4+'humidity'+'\n')
            for i in range(self.lat_dim):
                for j in range(self.long_dim):
                    for k in range(self.height_dim):
                        temp = random.uniform(-20, 50)
                        humid = random.uniform(0, 100)
                        file.write(str(i)+'\t\t'+str(j) + '\t\t'+str(k) +
                                   '\t\t'+str(temp)+'\t\t'+str(humid)+'\n')
        return None

    def __generate_data_4D(self) -> None:  # Generates 4-Dimensional Random Dataset
        random.seed(self.seed)
        with open(self.fileNameTXT, "w") as file:
            file.write('lat' + '\t'*2+'long' + '\t'+'height' + '\t'+'time' +
                       '\t'+'temprature'+'\t'*4+'humidity'+'\n')
            for i in range(self.lat_dim):
                for j in range(self.long_dim):
                    for k in range(self.height_dim):
                        for m in range(self.time_dim):
                            temp = random.uniform(-20, 50)
                            humid = random.uniform(0, 100)
                            file.write(str(i)+'\t\t'+str(j) + '\t\t'+str(k) + '\t\t'+str(m) +
                                       '\t\t'+str(temp)+'\t\t'+str(humid)+'\n')
        return None

    def generate_encryptData_from_txt(self, fileNameEnTXT, method='AES') -> None:
        
        if method=='AES_XTS':
            crypto_obj = AES_XTS_crypto()
        
        delimiter = '\t\t'
        with open(fileNameEnTXT, "w") as f:
            f.write('lat' + '\t'*2+'long' + '\t'+'height' + '\t'+'time' +
                                '\t'+'temprature'+'\t'*4+'humidity'+'\n')
            
            with open(self.fileNameTXT, 'r') as f2:
                line = f2.readline()
                
                for line_num in tqdm(range( np.prod(self.dataShape[:-1]) )):
                    
                    line = f2.readline().strip().split(delimiter)
                    
                    if method=='AES':
                        crypto_obj = AES_crypto()
                        humid=crypto_obj.AES_encrypt(line[-1])

                        crypto_obj = AES_crypto()
                        temp=crypto_obj.AES_encrypt(line[-2])
                    elif method=='AES_XTS':
                        humid=crypto_obj.AES_XTS_encrypt(line[-1], line_num+2)
                        temp=crypto_obj.AES_XTS_encrypt(line[-2], line_num+2)                       

                    line = [int(float(j)) for j in line[:-2]]
                    i,j,k,m=line[0],line[1],line[2],line[3]
                    f.write(str(i)+'\t\t'+str(j) + '\t\t'+str(k) + '\t\t'+str(m) +
                                '\t\t'+str(temp)+'\t\t'+str(humid)+'\n')
        
        logging.info(f'The {fileNameEnTXT} generated!')    

def create_kdtree(data_dim, fileNameTX, kdtree_dir, method):
    delimiter = "\t\t"
    data = []
    for i in range(2, np.prod(data_dim[1:])+2):
        x = linecache.getline(
            fileNameTX, i).strip().split(delimiter)
        x = x[:-2]
        x = [int(j) for j in x]
        data.append(x)

    kdtree = KdTree(kdtree_dir)
    kdtree.build(data)
    logging.info(f'The {kdtree_dir} generated!')

# kdtree form slab level shuffling
def create_kdtree_slab_shuffled(data_dim, fileNameTX, kdtree_dir, slab_size):
    delimiter = "\t\t"
    data = []
    for i in range(2, np.prod(data_dim[1:])+2, slab_size):
        x = linecache.getline(
            fileNameTX, i).strip().split(delimiter)
        x = x[:-2]
        x = [int(j) for j in x]
        data.append(x)
    kdtree = KdTree(kdtree_dir)
    kdtree.build(data)
    logging.info(f'The {kdtree_dir} generated!')


def create_kdtree_slab_matrix_shuffled(data_dim, fileNameTX, kdtree_dir, slab_dim):
    delimiter = "\t\t"
    data = []
    slab_numbers = int(np.prod(data_dim[1:]) / np.prod(slab_dim))
    slab_index = tuple([0]*len(slab_dim))
    for i in range(1, slab_numbers+1):

        i_org = slab_matrix2numbers(tuple(data_dim[1:]), slab_dim, i, slab_index)

        x = linecache.getline(fileNameTX, i_org+1).strip().split(delimiter)
        x = x[:-2]
        x = [int(j) for j in x]
        data.append(x)
    kdtree = KdTree(kdtree_dir)
    kdtree.build(data)
    logging.info(f'The {kdtree_dir} generated!')