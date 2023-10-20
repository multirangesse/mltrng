import linecache
import math
import pickle
import gc

import numpy as np

from query import query
from crypto.encryption import AES_crypto, AES_XTS_crypto
from kdtree.kdtree import KdTree
from utils.utils import event_duration, number2index, index2number
from collections import defaultdict


class dataload:
    def __init__(self,
                 dir_txt,
                 data_dim=2,
                 lat_dim=100,
                 long_dim=100,
                 height_dim=None,
                 time_dim=None,
                 delimiter="\t\t",
                 encrypted=None,
                 shuffled=None,
                 map_dir=None,
                 line_2_ijkm_dir=None,
                 ) -> None:
        self.data_dim = data_dim
        self.lat_dim = lat_dim
        self.long_dim = long_dim
        self.height_dim = height_dim
        self.time_dim = time_dim
        self.fileNameTXT = dir_txt
        self.delimiter = delimiter
        self.encrypted = encrypted
        self.shuffled = shuffled
        self.map_dir = map_dir
        self.line_2_ijkm_dir = line_2_ijkm_dir
        self.qrtime = 0
        self.buckets_ctr = defaultdict(int)
        self.bucket_num = 1
        self.line_size = 103
        self.bucket_dim = [3,3,3,3]
        self.bucket_size = np.prod(self.bucket_dim)*self.line_size
        self.part_nums = 0

        if self.data_dim == 2:
            self.__I = (self.lat_dim, self.long_dim)
        elif self.data_dim == 3:
            self.__I = (self.lat_dim, self.long_dim, self.height_dim)
        elif self.data_dim == 4:
            self.__I = (self.lat_dim, self.long_dim,
                        self.height_dim, self.time_dim)

        if self.encrypted == 'AES':
            self.crypto_obj = AES_crypto()
        elif self.encrypted == 'AES_XTS':
            self.crypto_obj = AES_XTS_crypto()

           
    def load_slabs(self, i_start, i_finish, map_dir=None):
        self.buckets_ctr = defaultdict(int)
        ed = event_duration(scale='milli', log=False)
        ed.event_start()
        if self.shuffled == 'index_0':
            rangeQuery = query.query_2_range_shuffled(
                i_start, i_finish, map_dir)
        elif self.shuffled == 'index_1':
            rangeQuery = query.query_2_range_shuffled_index_1(
                i_start, i_finish, map_dir)
        elif self.shuffled == 'index_2':
            rangeQuery = query.query_2_range_shuffled_index_2(
                i_start, i_finish, map_dir)
        else:
            rangeQuery = query.query_2_range(i_start, i_finish)           
        dataSlabs = []
        for slab_indx in rangeQuery:
            dataSlabs.append(self.load_slab(slab_indx[0], slab_indx[1]))

        dataSlabs = np.array(dataSlabs)
        dataSlabs = dataSlabs.reshape(-1, self.data_dim+2)
        time=ed.event_finish()
        # print(self.buckets_ctr) 
        return dataSlabs,time

    def load_slab(self, i_start, i_finish):       
        lines = []
        start_line = self.__ijkm_2_line(i_start)
        finish_line = self.__ijkm_2_line(i_finish)

        if self.shuffled == 'element':
            with open(self.map_dir, 'rb') as handle:
                map_pos = pickle.load(handle)
        elif self.shuffled == 'index_0':
            with open(self.map_dir, 'rb') as handle:
                map_pos = pickle.load(handle)
            map_pos_reverse = {val: key for key, val in map_pos.items()}
            with open(self.line_2_ijkm_dir, 'rb') as handle:
                line_2_ijkm = pickle.load(handle)
        elif self.shuffled == 'index_1':
            with open(self.map_dir, 'rb') as handle:
                map_pos = pickle.load(handle)
            map_pos_reverse = {val: key for key, val in map_pos.items()}
            with open(self.line_2_ijkm_dir, 'rb') as handle:
                line_2_ijkm = pickle.load(handle)
        elif self.shuffled == 'index_2':
            with open(self.map_dir, 'rb') as handle:
                map_pos = pickle.load(handle)
            map_pos_reverse = {val: key for key, val in map_pos.items()}
            with open(self.line_2_ijkm_dir, 'rb') as handle:
                line_2_ijkm = pickle.load(handle)
        for i in range(start_line, finish_line):
            # i is the old line number in original file
            # ii is the new line number in shuffled file
            ii = i
            if self.shuffled == 'element':
                ii = map_pos[i]
            elif self.shuffled == 'index_0':
                iii = line_2_ijkm[i]
                i = self.__ijkm_2_line(
                    (map_pos_reverse[iii[0]], iii[1], iii[2], iii[3]))
            elif self.shuffled == 'index_1':
                iii = line_2_ijkm[i]
                i = self.__ijkm_2_line(
                    (iii[0], map_pos_reverse[iii[1]], iii[2], iii[3]))
            elif self.shuffled == 'index_2':
                iii = line_2_ijkm[i]
                i = self.__ijkm_2_line(
                    (iii[0], iii[1], map_pos_reverse[iii[2]], iii[3]))
                
            self.buckets_ctr[line2bucket(ii, self.line_size, self.bucket_size)] += 1                   
            x = linecache.getline(
                self.fileNameTXT, ii).strip().split(self.delimiter)

            if self.encrypted == 'AES':
                x[:-2] = [float(j) for j in x[:-2]]
                x[-1] = self.crypto_obj.AES_decrypt(x[-1])
                x[-2] = self.crypto_obj.AES_decrypt(x[-2])
            elif self.encrypted == 'AES_XTS':
                x[:-2] = [float(j) for j in x[:-2]]
                x[-1] = self.crypto_obj.AES_XTS_decrypt(x[-1], i)
                x[-2] = self.crypto_obj.AES_XTS_decrypt(x[-2], i)
            elif self.encrypted == None:
                x = [float(j) for j in x]

            lines.append(x)           
        return lines 

    # !!! only loads data generated by generate_data()
    def load_file_txt(self):
        lines = []
        for i in range(self.start_line, self.start_line+self.chunk_size):
            x = linecache.getline(
                self.fileNameTXT, i).strip().split(self.delimiter)
            x = [float(j) for j in x]
            lines.append(x)
        return [i[self.data_dim:] for i in lines]

    # Maps the 4-dimension ranges to line numbers

    def __ijkm_2_line(self, i):
        dim = len(i)
        if dim == 2:
            return i[0]*self.__I[1]+i[1]+2
        elif dim == 3:
            return i[0]*self.__I[1]*self.__I[2] + i[1]*self.__I[2] + i[2]+2
        elif dim == 4:
            return i[0]*self.__I[1]*self.__I[2]*self.__I[3] + i[1]*self.__I[2]*self.__I[3] + i[2]*self.__I[3] + i[3]+2


    # Range Query on Kdtree
    def range_query_kdtree(self, r_start, r_finish, kdtree_filePath, method='disk'):
        self.buckets_ctr = defaultdict(int)        
        ed = event_duration(scale='milli', log=False)
        if method=='ram':
            with open(kdtree_filePath, 'rb') as handle:
                    kdtree = pickle.load(handle)
        elif method=='disk':
            kdtree=KdTree(kdtree_filePath)

        if self.shuffled == 'element':
            with open(self.map_dir, 'rb') as handle:
                map_pos = pickle.load(handle)
            map_pos_reverse = {val: key for key, val in map_pos.items()}        
        
        ed.event_start()
        area = (r_start[0],r_finish[0]-1,r_start[1],r_finish[1]-1,r_start[2],r_finish[2]-1,r_start[3],r_finish[3]-1)
        query_result_indxs=kdtree.query(area)
        query_result=[]
        for i in range(len(query_result_indxs)):
            self.buckets_ctr[line2bucket(query_result_indxs[i]+2, self.line_size, self.bucket_size)] += 1
            x = linecache.getline(
                    self.fileNameTXT, query_result_indxs[i]+2).strip().split(self.delimiter)

            if self.shuffled == 'element':
                location=map_pos_reverse[query_result_indxs[i]+2]
            else:
                location=query_result_indxs[i]+2

            if self.encrypted == 'AES':
                x[:-2] = [float(j) for j in x[:-2]]
                x[-1] = self.crypto_obj.AES_decrypt(x[-1])
                x[-2] = self.crypto_obj.AES_decrypt(x[-2])
            elif self.encrypted == 'AES_XTS':
                x[:-2] = [float(j) for j in x[:-2]]
                x[-1] = self.crypto_obj.AES_XTS_decrypt(x[-1], location)
                x[-2] = self.crypto_obj.AES_XTS_decrypt(x[-2], location)
            elif self.encrypted == None:
                x = [float(j) for j in x]

            query_result.append(x)

        query_result = np.array(query_result)
        query_result = query_result.reshape(-1, self.data_dim+2)	
        
        time=ed.event_finish() 
        return query_result,time

    # Range Query on Kdtree_slab shuffling
    def range_query_kdtree_slab(self, r_start, r_finish, kdtree_filePath, slab_size):

        data_dim = tuple(self.__I)

        self.buckets_ctr = defaultdict(int) 
        ed = event_duration(scale='milli', log=False)
        kdtree=KdTree(kdtree_filePath)

        with open(self.map_dir, 'rb') as handle:
            map_pos = pickle.load(handle)
        map_pos_reverse = {val: key for key, val in map_pos.items()}        
        
        ed.event_start()
        area = (r_start[0],r_finish[0]-1,r_start[1],r_finish[1]-1,r_start[2],r_finish[2]-1,r_start[3],r_finish[3]-1)
        area = area2slab_area(data_dim, slab_size, area)
        query_result_indxs = kdtree.query(area)

        # part numbers
        self.part_nums = len(query_result_indxs)

        # slab 2 lines
        new_line_list = []
        old_line_list = []
        for i in query_result_indxs:
            new_lines = slab2lines(i, slab_size)
            [new_line_list.append(ii) for ii in new_lines]
            old_line = map_pos_reverse[slab2lines(i,slab_size)[0]]
            old_lines = list(range(old_line, old_line+slab_size))
            [old_line_list.append(ii) for ii in old_lines]

        query_result=[]
        for new_line, old_line in zip(new_line_list, old_line_list):
            self.buckets_ctr[line2bucket(new_line, self.line_size, self.bucket_size)] += 1
            x = linecache.getline(self.fileNameTXT, new_line).strip().split("\t\t")        
            
            if self.encrypted == 'AES':
                x[:-2] = [float(j) for j in x[:-2]]
                x[-1] = self.crypto_obj.AES_decrypt(x[-1])
                x[-2] = self.crypto_obj.AES_decrypt(x[-2])
            elif self.encrypted == 'AES_XTS':
                x[:-2] = [float(j) for j in x[:-2]]
                x[-1] = self.crypto_obj.AES_XTS_decrypt(x[-1], old_line)
                x[-2] = self.crypto_obj.AES_XTS_decrypt(x[-2], old_line)
            elif self.encrypted == None:
                x = [float(j) for j in x]           

            query_result.append(x)

        query_result_final = []
        for point in query_result:
            if compare_area_point(r_start, r_finish, point[0:4]):
                query_result_final.append(point)

        query_result_final = np.array(query_result_final)
        query_result_final = query_result_final.reshape(-1, self.data_dim+2)	
        
        # print(len(query_result), len(query_result_final))

        time = ed.event_finish()
        # print(self.buckets_ctr) 
        return query_result_final, time
    

    # Range Query on Kdtree_slab_matrix shuffling
    def range_query_kdtree_slab_matrix(self, r_start, r_finish, kdtree_filePath, slab_dim, data_dim):
        
        data_dim = tuple(data_dim[1:])
        
        self.buckets_ctr = defaultdict(int) 
        ed = event_duration(scale='milli', log=False)
        kdtree=KdTree(kdtree_filePath)

        with open(self.map_dir, 'rb') as handle:
            map_pos = pickle.load(handle)
        map_pos_reverse = {val: key for key, val in map_pos.items()}        
        
        ed.event_start()
        area = (r_start[0],r_finish[0]-1,r_start[1],r_finish[1]-1,r_start[2],r_finish[2]-1,r_start[3],r_finish[3]-1)
        area = area2slab_matrix_area(data_dim, slab_dim, area)
        query_result_indxs=kdtree.query(area)

        # part numbers
        self.part_nums = len(query_result_indxs)

        # slab 2 lines
        new_line_list = []
        old_line_list = []
        for slab_num in query_result_indxs:
            new_lines = slab_matrix2numbers(data_dim, slab_dim, slab_num+1)
            [new_line_list.append(ii+1) for ii in new_lines]
            slab_num_org = map_pos_reverse[slab_num+1]
            old_lines = slab_matrix2numbers(data_dim, slab_dim, slab_num_org)
            [old_line_list.append(ii+1) for ii in old_lines]

        query_result=[]
        for new_line, old_line in zip(new_line_list, old_line_list):
            self.buckets_ctr[ line2slab_matrix_bucket(data_dim, self.bucket_dim, new_line) ] += 1
            x = linecache.getline(self.fileNameTXT, new_line).strip().split("\t\t")

            if self.encrypted == 'AES':
                x[:-2] = [float(j) for j in x[:-2]]
                x[-1] = self.crypto_obj.AES_decrypt(x[-1])
                x[-2] = self.crypto_obj.AES_decrypt(x[-2])
            elif self.encrypted == 'AES_XTS':
                x[:-2] = [float(j) for j in x[:-2]]
                x[-1] = self.crypto_obj.AES_XTS_decrypt(x[-1], old_line)
                x[-2] = self.crypto_obj.AES_XTS_decrypt(x[-2], old_line)
            elif self.encrypted == None:
                x = [float(j) for j in x]             

            query_result.append(x)

        query_result_final = []
        for point in query_result:
            if compare_area_point(r_start, r_finish, point[0:4]):
                query_result_final.append(point)

        query_result_final = np.array(query_result_final)
        query_result_final = query_result_final.reshape(-1, self.data_dim+2)	
        
        # print(len(query_result), len(query_result_final))

        time = ed.event_finish()
        # print(self.buckets_ctr) 
        return query_result_final, time






def compare_area_point(i_start, i_finish, point):
    cn1 = np.array(point, dtype=int)  >= np.array(i_start, dtype=int)
    cn2 = np.array(point, dtype=int)  < np.array(i_finish, dtype=int)
    return all(cn1 & cn2)

def slab2lines(slab_index, slab_size):
    return list(range(slab_index*slab_size+2, slab_index*slab_size+2+slab_size))

def line2bucket(line_number, line_size, bucket_size):
    return int((line_number-1)*line_size/bucket_size)+1

def slab_matrix2numbers(dim, slab_dim, slab_number, slab_index=None):
    slab_set_dim = tuple([int(dim[i]/slab_dim[i]) for i in range(len(dim))])
    min_s = [i*j for i,j in zip(number2index(slab_set_dim, slab_number), slab_dim)]
    if slab_index!=None:
        location = tuple([i+j for i,j in zip(min_s, slab_index)])
        location = index2number(dim , location)
        return location
    else:
        max_s = [i*j+j for i,j in zip(number2index(slab_set_dim, slab_number), slab_dim)]
        index_range = []
        for i,j in zip(min_s, max_s):
            index_range.append(f'{i}:{j}:1')
        index_range = ','.join(index_range)
        slab_numbers = eval(f'np.mgrid[{index_range}].reshape({len(dim)},-1).T')
        slab_numbers = [index2number(dim, i) for i in slab_numbers]
        return slab_numbers

def number2slab_matrix(dim, slab_dim, number):
    slab_set_dim = tuple([int(dim[i]/slab_dim[i]) for i in range(len(dim))])
    location = number2index(dim, number)
    slab_index = [0]*len(dim)
    min_s = [0]*len(dim)
    for i in range(len(dim)):
        min_s[i] = location[i]//slab_dim[i]
        slab_index[i] = location[i]%slab_dim[i]
    return index2number(slab_set_dim , min_s), slab_index


def line2slab(slab_size, line_number):
    return (line_number-2) // slab_size

def area2slab_area(data_dim, slab_size, area):
    slab_list = []
    for i in area[0:2]:
        for j in area[2:4]:
            for k in area[4:6]:
                for m in area[6:8]:
                    num = index2number(data_dim, (i,j,k,m)) + 1
                    slab_list.append( line2slab(slab_size, num) )
    slab_list = sorted(list(set(slab_list)))

    slab_range = []
    for i in slab_list:
        slab_start  = slab2lines(i, slab_size)[0]
        slab_range.append( number2index(data_dim, slab_start-1) )

    slab_range = np.array(slab_range)
    slab_range = [[min(slab_range[:,i]), max(slab_range[:,i])] for i in range(len(data_dim))]
    slab_range = np.array(slab_range)
    return tuple(slab_range.flatten())



def area2slab_matrix_area(data_dim, slab_dim, area):
    slab_list = []
    for i in area[0:2]:
        for j in area[2:4]:
            for k in area[4:6]:
                for m in area[6:8]:
                    num = index2number(data_dim, (i,j,k,m)) 
                    slab_list.append( number2slab_matrix(data_dim, slab_dim, num)[0] )  
    
    slab_list = sorted(list(set(slab_list)))
    slab_range = []
    for i in slab_list:
        slab_start  = slab_matrix2numbers(data_dim, slab_dim, i, slab_index=None)[0] + 1
        slab_range.append( number2index(data_dim, slab_start-1) )

    slab_range = np.array(slab_range)
    slab_range = [[min(slab_range[:,i]), max(slab_range[:,i])] for i in range(len(data_dim))]
    slab_range = np.array(slab_range)
    return tuple(slab_range.flatten())


def line2slab_matrix_bucket(dim, bucket_dim, line_num):
    slab_set_dim = tuple([math.ceil(dim[i]/bucket_dim[i]) for i in range(len(dim))])
    location = number2index(dim, line_num-1)
    min_s = [0]*len(dim)
    for i in range(len(dim)):
        min_s[i] = location[i]//bucket_dim[i]
    return index2number(slab_set_dim , min_s)