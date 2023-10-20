import argparse
import os
import logging

import numpy as np

from utils.welcome import greeting
from utils.report import report
from utils.utils import event_duration, query_yes_no_exe, query_yes_no_exe_hdf5, path, query_report
from datagen.datagent import datagen, datagenHdf5, create_kdtree, create_kdtree_slab_shuffled, create_kdtree_slab_matrix_shuffled
from dataloader.dataloader import dataload
from crypto import permute

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)") 
greeting()


ed = event_duration(scale='milli', log=True)


parser = argparse.ArgumentParser()
parser.add_argument('-dg', '--datagen', default= None, choices=['txt', 'hdf5', 'kdtree'])
parser.add_argument('-en', '--encrypt', action=argparse.BooleanOptionalAction)
parser.add_argument('-kd', '--kdisk', action=argparse.BooleanOptionalAction)
parser.add_argument('-sh', '--shuffle', default= None, choices=['sh', 'sh0', 'sh1', 'sh2', 'shs', 'shsm'])
parser.add_argument('-re', '--report', action=argparse.BooleanOptionalAction)
parser.add_argument('-te', '--test', action=argparse.BooleanOptionalAction)
parser.add_argument('-dd', '--datadim', type= int, nargs=5, required= False, default= [4, 10, 10, 10, 10])
parser.add_argument('-sd', '--slabdim', type= int, nargs=4, required= False, default= [2,2,2,2])
parser.add_argument('-ss', '--slab_size', type= int, required= False, default= 16)

parser.add_argument('-qu', '--query', default= None, choices=['txt', 'hdf5', 'kdtree'])
parser.add_argument('-rq', '--rangequery', type= int, nargs=8, required= False, default= [0, 5, 1, 6, 0, 2, 0, 10])


args = parser.parse_args()
ds_size = np.prod(args.datadim[1:])


def datagen_case(dg, en, sh, kd):
    command = [dg, en, sh, kd]
    if 'hdf5' in command:
        dataGenerator_hdf5 = datagenHdf5(path.hdf5_org,
                                    data_dim=args.datadim[0],
                                    lat_dim=args.datadim[1],
                                    long_dim=args.datadim[2],
                                    height_dim=args.datadim[3],
                                    time_dim=args.datadim[4],)    
    elif 'txt' in command:
        dataGenerator_org = datagen(path.org_data,
                                data_dim=args.datadim[0],
                                lat_dim=args.datadim[1],
                                long_dim=args.datadim[2],
                                height_dim=args.datadim[3],
                                time_dim=args.datadim[4])    
    match command:
        case ['hdf5', None, None, None]:
            query_yes_no_exe_hdf5(path.org_data, path.hdf5_org, 
                                  dataGenerator_hdf5.generate_data_from_txt, 
                                  path.hdf5_org, path.org_data)
        case ['hdf5', True, None, None]:
            query_yes_no_exe_hdf5(path.org_encrypted, path.hdf5_encrypted, 
                                  dataGenerator_hdf5.generate_data_from_txt, 
                                  path.hdf5_encrypted, path.org_encrypted)
        case ['hdf5', True, 'sh', None]:
            query_yes_no_exe_hdf5(path.encrypted_shuffled, path.hdf5_encrypted_shuffled, 
                                    dataGenerator_hdf5.generate_data_from_txt, 
                                    path.hdf5_encrypted_shuffled, path.encrypted_shuffled)
        case ['hdf5', True, 'sh0', None]:
            query_yes_no_exe_hdf5(path.encrypted_shuffled_index0, path.hdf5_encrypted_shuffled_index0, 
                                    dataGenerator_hdf5.generate_data_from_txt, 
                                    path.hdf5_encrypted_shuffled_index0, path.encrypted_shuffled_index0)
        case ['hdf5', True, 'sh1', None]:
            query_yes_no_exe_hdf5(path.encrypted_shuffled_index1, path.hdf5_encrypted_shuffled_index1, 
                                    dataGenerator_hdf5.generate_data_from_txt, 
                                    path.hdf5_encrypted_shuffled_index1, path.encrypted_shuffled_index1)
        case ['hdf5', True, 'sh2', None]:
            query_yes_no_exe_hdf5(path.encrypted_shuffled_index2, path.hdf5_encrypted_shuffled_index2, 
                                    dataGenerator_hdf5.generate_data_from_txt, 
                                    path.hdf5_encrypted_shuffled_index2, path.encrypted_shuffled_index2)
        case ['txt', None, None, None]:
            query_yes_no_exe(path.org_data, dataGenerator_org.generate_data)
        case ['txt', True, None, None]:
            query_yes_no_exe(path.org_encrypted, 
                             dataGenerator_org.generate_encryptData_from_txt, path.org_encrypted, 'AES_XTS')
        case ['txt', True, 'sh', None]:
            ds_pos=list(range(2,ds_size+2))
            query_yes_no_exe(path.mapping, permute.prf, ds_pos, 
                                path.mapping, args.datadim, path.line_2_ijkm)
            query_yes_no_exe(path.encrypted_shuffled, permute.permute, 
                                path.org_encrypted, path.encrypted_shuffled, path.mapping)
        case ['txt', True, 'sh0', None]:
            query_yes_no_exe(path.mapping_index0, permute.prf_block, path.mapping_index0, 
                                args.datadim, 1)
            query_yes_no_exe(path.encrypted_shuffled_index0, permute.permute_indx_0, 
                                path.org_encrypted, path.encrypted_shuffled_index0, 
                                path.mapping_index0, args.datadim)
        case ['txt', True, 'sh1', None]:
            query_yes_no_exe(path.mapping_index1, permute.prf_block, path.mapping_index1, 
                                args.datadim, 2)
            query_yes_no_exe(path.encrypted_shuffled_index1, permute.permute_indx_1, 
                                path.org_encrypted, path.encrypted_shuffled_index1, 
                                path.mapping_index1, args.datadim)
        case ['txt', True, 'sh2', None]:
            query_yes_no_exe(path.mapping_index2, permute.prf_block, path.mapping_index2, 
                                args.datadim, 3)
            query_yes_no_exe(path.encrypted_shuffled_index2, permute.permute_indx_2, 
                                path.org_encrypted, path.encrypted_shuffled_index2, 
                                path.mapping_index2, args.datadim)
        case ['txt', True, 'shs', None]:
            query_yes_no_exe(path.mapping_slab, permute.prf_slab, args.slab_size, ds_size, path.mapping_slab)
            query_yes_no_exe(path.encrypted_shuffled_slab, permute.permute_slab,
                                path.org_encrypted, path.encrypted_shuffled_slab, 
                                path.mapping_slab, args.slab_size)
        case ['txt', True, 'shsm', None]:
            slab_set_dim = tuple([int(args.datadim[i+1]/args.slabdim[i]) for i in range(len(args.datadim[1:]))])
            org_data = list(range(1,np.prod(slab_set_dim)+1))
            query_yes_no_exe(path.mapping_slab_matrix, permute.prf_general, org_data, path.mapping_slab_matrix)
            query_yes_no_exe(path.encrypted_shuffled_slab_matrix, permute.permute_slab_matrix,
                                path.org_encrypted, path.encrypted_shuffled_slab_matrix, 
                                path.mapping_slab_matrix, args.slabdim, args.datadim)
        case ['kdtree', True, None, None]:
            query_yes_no_exe(path.kdtree_encrypted_obj, create_kdtree, args.datadim, 
                             path.org_encrypted, path.kdtree_encrypted_obj, 'memory')
        case ['kdtree', True, None, True]:
            query_yes_no_exe(path.kdtree_encrypted, create_kdtree, args.datadim, 
                             path.org_encrypted, path.kdtree_encrypted, 'disk')
        case ['kdtree', True, 'sh', True]:
                query_yes_no_exe(path.kdtree_encrypted_shuffled, create_kdtree, args.datadim, 
                                 path.encrypted_shuffled, path.kdtree_encrypted_shuffled, 'disk')
        case ['kdtree', True, 'shs', True]:
                query_yes_no_exe(path.kdtree_encrypted_shuffled_slab, create_kdtree_slab_shuffled, args.datadim, 
                                 path.encrypted_shuffled_slab, path.kdtree_encrypted_shuffled_slab, args.slab_size)
        case ['kdtree', True, 'shsm', True]: # data_dim, fileNameTX, kdtree_dir, slab_dim
                query_yes_no_exe(path.kdtree_encrypted_shuffled_slab_matrix, create_kdtree_slab_matrix_shuffled, args.datadim, 
                                 path.encrypted_shuffled_slab_matrix, path.kdtree_encrypted_shuffled_slab_matrix, args.slabdim)        
        case ['kdtree', True, 'sh', None]:
                query_yes_no_exe(path.kdtree_encrypted_shuffled_obj, create_kdtree, 
                                 args.datadim, path.encrypted_shuffled, 
                                 path.kdtree_encrypted_shuffled_obj, 'memory')
        case _:
            print('This case has not been implemented!')

def query_case(qu, en, sh, kd):
    command = [qu, en, sh, kd]

    i_start = tuple(args.rangequery[0::2])
    i_finish = tuple(args.rangequery[1::2])
    dataLoader = dataload(path.org_data,
                      data_dim=args.datadim[0],
                      lat_dim=args.datadim[1],
                      long_dim=args.datadim[2],
                      height_dim=args.datadim[3],
                      time_dim=args.datadim[4],
                      encrypted=None)
    queryResult, _ = query_report(args, ed, [], dataLoader.load_slabs, i_start, i_finish)
    queryResult = queryResult[0]

    match command:
        case ['txt', None, None, None]:
            _ = query_report(args, ed, queryResult, dataLoader.load_slabs, i_start, i_finish)
        case ['txt', True, None, None]:
            dataLoader = dataload(path.org_encrypted,
                      data_dim=args.datadim[0],
                      lat_dim=args.datadim[1],
                      long_dim=args.datadim[2],
                      height_dim=args.datadim[3],
                      time_dim=args.datadim[4],
                      encrypted='AES_XTS')
            _ = query_report(args, ed, queryResult, dataLoader.load_slabs, i_start, i_finish)
        case ['txt', True, 'sh', None]:
            dataLoader = dataload(path.encrypted_shuffled,
                    data_dim=args.datadim[0],
                    lat_dim=args.datadim[1],
                    long_dim=args.datadim[2],
                    height_dim=args.datadim[3],
                    time_dim=args.datadim[4],
                    encrypted='AES_XTS',
                    shuffled='element',
                    map_dir=path.mapping)
            _ = query_report(args, ed, queryResult, dataLoader.load_slabs, i_start, i_finish)
        case ['txt', True, 'sh0', None]:
            dataLoader = dataload(path.encrypted_shuffled_index0,
                    data_dim=args.datadim[0],
                    lat_dim=args.datadim[1],
                    long_dim=args.datadim[2],
                    height_dim=args.datadim[3],
                    time_dim=args.datadim[4],
                    encrypted='AES_XTS',
                    shuffled='index_0',
                    map_dir=path.mapping_index0,
                    line_2_ijkm_dir=path.line_2_ijkm)
            _ = query_report(args, ed, queryResult, dataLoader.load_slabs, i_start, i_finish, path.mapping_index0)
        case ['txt', True, 'sh1', None]:
            dataLoader = dataload(path.encrypted_shuffled_index1,
                    data_dim=args.datadim[0],
                    lat_dim=args.datadim[1],
                    long_dim=args.datadim[2],
                    height_dim=args.datadim[3],
                    time_dim=args.datadim[4],
                    encrypted='AES_XTS',
                    shuffled='index_1',
                    map_dir=path.mapping_index1,
                    line_2_ijkm_dir=path.line_2_ijkm)
            _ = query_report(args, ed, queryResult, dataLoader.load_slabs, i_start, i_finish, path.mapping_index1)
        case ['txt', True, 'sh2', None]:
            dataLoader = dataload(path.encrypted_shuffled_index2,
                    data_dim=args.datadim[0],
                    lat_dim=args.datadim[1],
                    long_dim=args.datadim[2],
                    height_dim=args.datadim[3],
                    time_dim=args.datadim[4],
                    encrypted='AES_XTS',
                    shuffled='index_2',
                    map_dir=path.mapping_index2,
                    line_2_ijkm_dir=path.line_2_ijkm)
            _ = query_report(args, ed, queryResult, dataLoader.load_slabs, i_start, i_finish, path.mapping_index2)
        case ['txt', True, 'shs', None]:
            print('This case has not been implemented!')
            # dataLoader = dataload(path.encrypted_shuffled_sl,
            #         data_dim=args.datadim[0],
            #         lat_dim=args.datadim[1],
            #         long_dim=args.datadim[2],
            #         height_dim=args.datadim[3],
            #         time_dim=args.datadim[4],
            #         encrypted='AES_XTS',
            #         shuffled='index_2',
            #         map_dir=path.mapping_index2,
            #         line_2_ijkm_dir=path.line_2_ijkm)
            # _ = query_report(args, ed, queryResult, dataLoader.load_slabs, i_start, i_finish, path.mapping_index2)
        case ['hdf5', None, None, None]:
            _ = query_report(args, ed, queryResult, dataLoader.range_query_HDF5, i_start, i_finish, path.hdf5_org)
        case ['hdf5', True, None, None]:
            dataLoader = dataload(path.org_encrypted,
                      data_dim=args.datadim[0],
                      lat_dim=args.datadim[1],
                      long_dim=args.datadim[2],
                      height_dim=args.datadim[3],
                      time_dim=args.datadim[4],
                      encrypted='AES_XTS')
            _ = query_report(args, ed, queryResult, dataLoader.range_query_HDF5, i_start, i_finish, path.hdf5_encrypted)
        case ['hdf5', True, 'sh', None]:
            dataLoader = dataload(path.encrypted_shuffled,
                    data_dim=args.datadim[0],
                    lat_dim=args.datadim[1],
                    long_dim=args.datadim[2],
                    height_dim=args.datadim[3],
                    time_dim=args.datadim[4],
                    encrypted='AES_XTS',
                    shuffled='element',
                    map_dir=path.mapping,
                    line_2_ijkm_dir=path.line_2_ijkm)
            _ = query_report(args, ed, queryResult, dataLoader.range_query_shuffled_HDF5, 
                                i_start, i_finish, path.hdf5_encrypted_shuffled)
        case ['hdf5', True, 'sh0', None]:
            dataLoader = dataload(path.encrypted_shuffled_index0,
                    data_dim=args.datadim[0],
                    lat_dim=args.datadim[1],
                    long_dim=args.datadim[2],
                    height_dim=args.datadim[3],
                    time_dim=args.datadim[4],
                    encrypted='AES_XTS',
                    shuffled='index_0',
                    map_dir=path.mapping_index0,
                    line_2_ijkm_dir=path.line_2_ijkm)
            _ = query_report(args, ed, queryResult, dataLoader.range_query_shuffled_dim_HDF5, 
                                i_start, i_finish, path.hdf5_encrypted_shuffled_index0)
        case ['hdf5', True, 'sh1', None]:
            dataLoader = dataload(path.encrypted_shuffled_index1,
                    data_dim=args.datadim[0],
                    lat_dim=args.datadim[1],
                    long_dim=args.datadim[2],
                    height_dim=args.datadim[3],
                    time_dim=args.datadim[4],
                    encrypted='AES_XTS',
                    shuffled='index_1',
                    map_dir=path.mapping_index1,
                    line_2_ijkm_dir=path.line_2_ijkm)
            _ = query_report(args, ed, queryResult, dataLoader.range_query_shuffled_dim_HDF5, 
                                i_start, i_finish, path.hdf5_encrypted_shuffled_index1)
        case ['hdf5', True, 'sh2', None]:
            dataLoader = dataload(path.encrypted_shuffled_index2,
                    data_dim=args.datadim[0],
                    lat_dim=args.datadim[1],
                    long_dim=args.datadim[2],
                    height_dim=args.datadim[3],
                    time_dim=args.datadim[4],
                    encrypted='AES_XTS',
                    shuffled='index_2',
                    map_dir=path.mapping_index2,
                    line_2_ijkm_dir=path.line_2_ijkm)
            _ = query_report(args, ed, queryResult, dataLoader.range_query_shuffled_dim_HDF5, 
                                i_start, i_finish, path.hdf5_encrypted_shuffled_index2)
        case ['kdtree', True, None, None]:
            dataLoader = dataload(path.org_encrypted,
                                data_dim=args.datadim[0],
                                lat_dim=args.datadim[1],
                                long_dim=args.datadim[2],
                                height_dim=args.datadim[3],
                                time_dim=args.datadim[4],
                                encrypted = 'AES_XTS')
            _ = query_report(args, ed, queryResult, dataLoader.range_query_kdtree, 
                                 i_start, i_finish, path.kdtree_encrypted_obj, 'ram')
        case ['kdtree', True, None, True]:
            dataLoader = dataload(path.org_encrypted,
                                data_dim=args.datadim[0],
                                lat_dim=args.datadim[1],
                                long_dim=args.datadim[2],
                                height_dim=args.datadim[3],
                                time_dim=args.datadim[4],
                                encrypted = 'AES_XTS')
            _ = query_report(args, ed, queryResult, dataLoader.range_query_kdtree, 
                                 i_start, i_finish, path.kdtree_encrypted, 'disk')
        case ['kdtree', True, 'sh', True]:
            dataLoader = dataload(path.encrypted_shuffled,
                                data_dim=args.datadim[0],
                                lat_dim=args.datadim[1],
                                long_dim=args.datadim[2],
                                height_dim=args.datadim[3],
                                time_dim=args.datadim[4],
                                encrypted = 'AES_XTS',
                                shuffled='element',
                                map_dir=path.mapping)
            _ = query_report(args, ed, queryResult, dataLoader.range_query_kdtree, 
                                i_start, i_finish, path.kdtree_encrypted_shuffled, 'disk')
        case ['kdtree', True, 'shs', True]:
            dataLoader = dataload(path.encrypted_shuffled_slab,
                                data_dim=args.datadim[0],
                                lat_dim=args.datadim[1],
                                long_dim=args.datadim[2],
                                height_dim=args.datadim[3],
                                time_dim=args.datadim[4],
                                encrypted = 'AES_XTS',
                                shuffled='slab',
                                map_dir=path.mapping_slab)
            _ = query_report(args, ed, queryResult, dataLoader.range_query_kdtree_slab, 
                                i_start, i_finish, path.kdtree_encrypted_shuffled_slab, args.slab_size)
        case ['kdtree', True, 'shsm', True]:
            dataLoader = dataload(path.encrypted_shuffled_slab_matrix,
                                data_dim=args.datadim[0],
                                lat_dim=args.datadim[1],
                                long_dim=args.datadim[2],
                                height_dim=args.datadim[3],
                                time_dim=args.datadim[4],
                                encrypted = 'AES_XTS',
                                shuffled='slab_matrix',
                                map_dir=path.mapping_slab_matrix)
            # range_query_kdtree_slab_matrix(self, r_start, r_finish, kdtree_filePath, slab_dim, data_dim)
            _ = query_report(args, ed, queryResult, dataLoader.range_query_kdtree_slab_matrix, 
                                i_start, i_finish, path.kdtree_encrypted_shuffled_slab_matrix, args.slabdim, args.datadim)        
        case ['kdtree', True, 'sh', None]:
            print('This case has not been implemented!')
        case _:
            print('This case has not been implemented!')


if args.datagen:
    datagen_case(args.datagen, args.encrypt, args.shuffle, args.kdisk)
if args.query:
    query_case(args.query, args.encrypt, args.shuffle, args.kdisk)
if args.report:
    report(args)

# if args.datagen:
#     if args.datagen=='hdf5':
#         dataGenerator_hdf5 = datagenHdf5(path.hdf5_org,
#                                     data_dim=args.datadim[0],
#                                     lat_dim=args.datadim[1],
#                                     long_dim=args.datadim[2],
#                                     height_dim=args.datadim[3],
#                                     time_dim=args.datadim[4],)
#         if args.encrypt==None and args.shuffle==None:
#             query_yes_no_exe_hdf5(path.org_data, path.hdf5_org, 
#                                   dataGenerator_hdf5.generate_data_from_txt, 
#                                   path.hdf5_org, path.org_data)
#         elif args.encrypt==True and args.shuffle==None:
#             query_yes_no_exe_hdf5(path.org_encrypted, path.hdf5_encrypted, 
#                                   dataGenerator_hdf5.generate_data_from_txt, 
#                                   path.hdf5_encrypted, path.org_encrypted)
#         elif args.encrypt==True and args.shuffle:
#             if args.shuffle=='sh':
#                 query_yes_no_exe_hdf5(path.encrypted_shuffled, path.hdf5_encrypted_shuffled, 
#                                       dataGenerator_hdf5.generate_data_from_txt, 
#                                       path.hdf5_encrypted_shuffled, path.encrypted_shuffled)
#             elif args.shuffle=='sh0':
#                 query_yes_no_exe_hdf5(path.encrypted_shuffled_index0, path.hdf5_encrypted_shuffled_index0, 
#                                       dataGenerator_hdf5.generate_data_from_txt, 
#                                       path.hdf5_encrypted_shuffled_index0, path.encrypted_shuffled_index0)
#             elif args.shuffle=='sh1':
#                 query_yes_no_exe_hdf5(path.encrypted_shuffled_index1, path.hdf5_encrypted_shuffled_index1, 
#                                       dataGenerator_hdf5.generate_data_from_txt, 
#                                       path.hdf5_encrypted_shuffled_index1, path.encrypted_shuffled_index1)
#             elif args.shuffle=='sh2':
#                 query_yes_no_exe_hdf5(path.encrypted_shuffled_index2, path.hdf5_encrypted_shuffled_index2, 
#                                       dataGenerator_hdf5.generate_data_from_txt, 
#                                       path.hdf5_encrypted_shuffled_index2, path.encrypted_shuffled_index2)
#     elif args.datagen=='txt':
#         dataGenerator_org = datagen(path.org_data,
#                                 data_dim=args.datadim[0],
#                                 lat_dim=args.datadim[1],
#                                 long_dim=args.datadim[2],
#                                 height_dim=args.datadim[3],
#                                 time_dim=args.datadim[4])
#         if args.encrypt==None and args.shuffle==None:
#             query_yes_no_exe(path.org_data, dataGenerator_org.generate_data)
#         elif args.encrypt==True and args.shuffle==None:
#             query_yes_no_exe(path.org_encrypted, 
#                              dataGenerator_org.generate_encryptData_from_txt, path.org_encrypted, 'AES_XTS')
#         elif args.encrypt==True and args.shuffle:
#             if args.shuffle=='sh':
#                 ds_pos=list(range(2,ds_size+2))
#                 query_yes_no_exe(path.mapping, permute.prf, ds_pos, 
#                                  path.mapping, args.datadim, path.line_2_ijkm)
#                 query_yes_no_exe(path.encrypted_shuffled, permute.permute, 
#                                  path.org_encrypted, path.encrypted_shuffled, path.mapping)
#             elif args.shuffle=='sh0':
#                 query_yes_no_exe(path.mapping_index0, permute.prf_block, path.mapping_index0, 
#                                  args.datadim, 1)
#                 query_yes_no_exe(path.encrypted_shuffled_index0, permute.permute_indx_0, 
#                                  path.org_encrypted, path.encrypted_shuffled_index0, 
#                                  path.mapping_index0, args.datadim)
#             elif args.shuffle=='sh1':
#                 query_yes_no_exe(path.mapping_index1, permute.prf_block, path.mapping_index1, 
#                                  args.datadim, 2)
#                 query_yes_no_exe(path.encrypted_shuffled_index1, permute.permute_indx_1, 
#                                  path.org_encrypted, path.encrypted_shuffled_index1, 
#                                  path.mapping_index1, args.datadim)
#             elif args.shuffle=='sh2':
#                 query_yes_no_exe(path.mapping_index2, permute.prf_block, path.mapping_index2, 
#                                  args.datadim, 3)
#                 query_yes_no_exe(path.encrypted_shuffled_index2, permute.permute_indx_2, 
#                                  path.org_encrypted, path.encrypted_shuffled_index2, 
#                                  path.mapping_index2, args.datadim)
#             elif args.shuffle=='shs':
#                 query_yes_no_exe(path.mapping_slab, permute.prf_slab, args.slab_size, ds_size, path.mapping_slab)
#                 query_yes_no_exe(path.encrypted_shuffled_slab, permute.permute_slab,
#                                  path.org_encrypted, path.encrypted_shuffled_slab, 
#                                  path.mapping_slab, args.slab_size)
#     elif args.datagen=='kdtree' and args.encrypt==True:
#         if args.kdisk==True and args.shuffle==None:
#             query_yes_no_exe(path.kdtree_encrypted, create_kdtree, args.datadim, 
#                              path.org_encrypted, path.kdtree_encrypted, 'disk')
#         elif args.kdisk==None and args.shuffle==None:
#             query_yes_no_exe(path.kdtree_encrypted_obj, create_kdtree, args.datadim, 
#                              path.org_encrypted, path.kdtree_encrypted_obj, 'memory')
#         elif args.kdisk==True and args.shuffle:
#             if args.shuffle=='sh':
#                 query_yes_no_exe(path.kdtree_encrypted_shuffled, create_kdtree, args.datadim, 
#                                  path.encrypted_shuffled, path.kdtree_encrypted_shuffled, 'disk')
#             elif args.shuffle=='shs':
#                 query_yes_no_exe(path.kdtree_encrypted_shuffled_slab, create_kdtree_slab_shuffled, args.datadim, 
#                                  path.encrypted_shuffled_slab, path.kdtree_encrypted_shuffled_slab, args.slab_size)
#         elif args.kdisk==None and args.shuffle:
#             if args.shuffle=='sh':
#                 query_yes_no_exe(path.kdtree_encrypted_shuffled_obj, create_kdtree, 
#                                  args.datadim, path.encrypted_shuffled, 
#                                  path.kdtree_encrypted_shuffled_obj, 'memory')


# if args.query:
#     i_start = tuple(args.rangequery[0::2])
#     i_finish = tuple(args.rangequery[1::2])
#     dataLoader = dataload(path.org_data,
#                       data_dim=args.datadim[0],
#                       lat_dim=args.datadim[1],
#                       long_dim=args.datadim[2],
#                       height_dim=args.datadim[3],
#                       time_dim=args.datadim[4],
#                       encrypted=None)
#     queryResult, _ = query_report(args, ed, [], dataLoader.load_slabs, i_start, i_finish)
#     queryResult = queryResult[0]
#     if args.query=='txt':
#         if args.encrypt==None and args.shuffle==None:
#             _ = query_report(args, ed, queryResult, dataLoader.load_slabs, i_start, i_finish)
#         elif args.encrypt==True and args.shuffle==None:
#             dataLoader = dataload(path.org_encrypted,
#                       data_dim=args.datadim[0],
#                       lat_dim=args.datadim[1],
#                       long_dim=args.datadim[2],
#                       height_dim=args.datadim[3],
#                       time_dim=args.datadim[4],
#                       encrypted='AES_XTS')
#             _ = query_report(args, ed, queryResult, dataLoader.load_slabs, i_start, i_finish)
#         elif args.encrypt==True and args.shuffle:
#             if args.shuffle=='sh':
#                 dataLoader = dataload(path.encrypted_shuffled,
#                       data_dim=args.datadim[0],
#                       lat_dim=args.datadim[1],
#                       long_dim=args.datadim[2],
#                       height_dim=args.datadim[3],
#                       time_dim=args.datadim[4],
#                       encrypted='AES_XTS',
#                       shuffled='element',
#                       map_dir=path.mapping)
#                 _ = query_report(args, ed, queryResult, dataLoader.load_slabs, i_start, i_finish)
#             elif args.shuffle=='sh0':
#                 dataLoader = dataload(path.encrypted_shuffled_index0,
#                       data_dim=args.datadim[0],
#                       lat_dim=args.datadim[1],
#                       long_dim=args.datadim[2],
#                       height_dim=args.datadim[3],
#                       time_dim=args.datadim[4],
#                       encrypted='AES_XTS',
#                       shuffled='index_0',
#                       map_dir=path.mapping_index0,
#                       line_2_ijkm_dir=path.line_2_ijkm)
#                 _ = query_report(args, ed, queryResult, dataLoader.load_slabs, i_start, i_finish, path.mapping_index0)
#             elif args.shuffle=='sh1':
#                 dataLoader = dataload(path.encrypted_shuffled_index1,
#                       data_dim=args.datadim[0],
#                       lat_dim=args.datadim[1],
#                       long_dim=args.datadim[2],
#                       height_dim=args.datadim[3],
#                       time_dim=args.datadim[4],
#                       encrypted='AES_XTS',
#                       shuffled='index_1',
#                       map_dir=path.mapping_index1,
#                       line_2_ijkm_dir=path.line_2_ijkm)
#                 _ = query_report(args, ed, queryResult, dataLoader.load_slabs, i_start, i_finish, path.mapping_index1)
#             elif args.shuffle=='sh2':
#                 dataLoader = dataload(path.encrypted_shuffled_index2,
#                       data_dim=args.datadim[0],
#                       lat_dim=args.datadim[1],
#                       long_dim=args.datadim[2],
#                       height_dim=args.datadim[3],
#                       time_dim=args.datadim[4],
#                       encrypted='AES_XTS',
#                       shuffled='index_2',
#                       map_dir=path.mapping_index2,
#                       line_2_ijkm_dir=path.line_2_ijkm)
#                 _ = query_report(args, ed, queryResult, dataLoader.load_slabs, i_start, i_finish, path.mapping_index2)
# # slab shuffling query
#             # elif args.shuffle=='shs':
#             #     dataLoader = dataload(path.encrypted_shuffled_sl,
#             #           data_dim=args.datadim[0],
#             #           lat_dim=args.datadim[1],
#             #           long_dim=args.datadim[2],
#             #           height_dim=args.datadim[3],
#             #           time_dim=args.datadim[4],
#             #           encrypted='AES_XTS',
#             #           shuffled='index_2',
#             #           map_dir=path.mapping_index2,
#             #           line_2_ijkm_dir=path.line_2_ijkm)
#             #     _ = query_report(args, ed, queryResult, dataLoader.load_slabs, i_start, i_finish, path.mapping_index2)
#     if args.query=='hdf5':
#         if args.encrypt==None and args.shuffle==None:
#             _ = query_report(args, ed, queryResult, dataLoader.range_query_HDF5, i_start, i_finish, path.hdf5_org)
#         elif args.encrypt==True and args.shuffle==None:
#             dataLoader = dataload(path.org_encrypted,
#                       data_dim=args.datadim[0],
#                       lat_dim=args.datadim[1],
#                       long_dim=args.datadim[2],
#                       height_dim=args.datadim[3],
#                       time_dim=args.datadim[4],
#                       encrypted='AES_XTS')
#             _ = query_report(args, ed, queryResult, dataLoader.range_query_HDF5, i_start, i_finish, path.hdf5_encrypted)
#         elif args.encrypt==True and args.shuffle:
#             if args.shuffle=='sh':
#                 dataLoader = dataload(path.encrypted_shuffled,
#                       data_dim=args.datadim[0],
#                       lat_dim=args.datadim[1],
#                       long_dim=args.datadim[2],
#                       height_dim=args.datadim[3],
#                       time_dim=args.datadim[4],
#                       encrypted='AES_XTS',
#                       shuffled='element',
#                       map_dir=path.mapping,
#                       line_2_ijkm_dir=path.line_2_ijkm)
#                 _ = query_report(args, ed, queryResult, dataLoader.range_query_shuffled_HDF5, 
#                                  i_start, i_finish, path.hdf5_encrypted_shuffled)
#             elif args.shuffle=='sh0':
#                 dataLoader = dataload(path.encrypted_shuffled_index0,
#                       data_dim=args.datadim[0],
#                       lat_dim=args.datadim[1],
#                       long_dim=args.datadim[2],
#                       height_dim=args.datadim[3],
#                       time_dim=args.datadim[4],
#                       encrypted='AES_XTS',
#                       shuffled='index_0',
#                       map_dir=path.mapping_index0,
#                       line_2_ijkm_dir=path.line_2_ijkm)
#                 _ = query_report(args, ed, queryResult, dataLoader.range_query_shuffled_dim_HDF5, 
#                                  i_start, i_finish, path.hdf5_encrypted_shuffled_index0)
#             elif args.shuffle=='sh1':
#                 dataLoader = dataload(path.encrypted_shuffled_index1,
#                       data_dim=args.datadim[0],
#                       lat_dim=args.datadim[1],
#                       long_dim=args.datadim[2],
#                       height_dim=args.datadim[3],
#                       time_dim=args.datadim[4],
#                       encrypted='AES_XTS',
#                       shuffled='index_1',
#                       map_dir=path.mapping_index1,
#                       line_2_ijkm_dir=path.line_2_ijkm)
#                 _ = query_report(args, ed, queryResult, dataLoader.range_query_shuffled_dim_HDF5, 
#                                  i_start, i_finish, path.hdf5_encrypted_shuffled_index1)
#             elif args.shuffle=='sh2':
#                 dataLoader = dataload(path.encrypted_shuffled_index2,
#                       data_dim=args.datadim[0],
#                       lat_dim=args.datadim[1],
#                       long_dim=args.datadim[2],
#                       height_dim=args.datadim[3],
#                       time_dim=args.datadim[4],
#                       encrypted='AES_XTS',
#                       shuffled='index_2',
#                       map_dir=path.mapping_index2,
#                       line_2_ijkm_dir=path.line_2_ijkm)
#                 _ = query_report(args, ed, queryResult, dataLoader.range_query_shuffled_dim_HDF5, 
#                                  i_start, i_finish, path.hdf5_encrypted_shuffled_index2)
#     elif args.query=='kdtree' and args.encrypt==True:
#         if args.kdisk==True and args.shuffle==None:
#             dataLoader = dataload(path.org_encrypted,
#                                 data_dim=args.datadim[0],
#                                 lat_dim=args.datadim[1],
#                                 long_dim=args.datadim[2],
#                                 height_dim=args.datadim[3],
#                                 time_dim=args.datadim[4],
#                                 encrypted = 'AES_XTS')
#             _ = query_report(args, ed, queryResult, dataLoader.range_query_kdtree, 
#                                  i_start, i_finish, path.kdtree_encrypted, 'disk')
#         elif args.kdisk==None and args.shuffle==None:
#             dataLoader = dataload(path.org_encrypted,
#                                 data_dim=args.datadim[0],
#                                 lat_dim=args.datadim[1],
#                                 long_dim=args.datadim[2],
#                                 height_dim=args.datadim[3],
#                                 time_dim=args.datadim[4],
#                                 encrypted = 'AES_XTS')
#             _ = query_report(args, ed, queryResult, dataLoader.range_query_kdtree, 
#                                  i_start, i_finish, path.kdtree_encrypted_obj, 'ram')
#         elif args.kdisk==True and args.shuffle:
#             if args.shuffle=='sh':
#                 dataLoader = dataload(path.encrypted_shuffled,
#                                     data_dim=args.datadim[0],
#                                     lat_dim=args.datadim[1],
#                                     long_dim=args.datadim[2],
#                                     height_dim=args.datadim[3],
#                                     time_dim=args.datadim[4],
#                                     encrypted = 'AES_XTS',
#                                     shuffled='element',
#                                     map_dir=path.mapping)
#                 _ = query_report(args, ed, queryResult, dataLoader.range_query_kdtree, 
#                                     i_start, i_finish, path.kdtree_encrypted_shuffled, 'disk')
#             elif args.shuffle=='shs':
#                 dataLoader = dataload(path.encrypted_shuffled_slab,
#                                     data_dim=args.datadim[0],
#                                     lat_dim=args.datadim[1],
#                                     long_dim=args.datadim[2],
#                                     height_dim=args.datadim[3],
#                                     time_dim=args.datadim[4],
#                                     encrypted = 'AES_XTS',
#                                     shuffled='slab',
#                                     map_dir=path.mapping_slab)
#                 _ = query_report(args, ed, queryResult, dataLoader.range_query_kdtree_slab, 
#                                     i_start, i_finish, path.kdtree_encrypted_shuffled_slab, args.slab_size)
#         elif args.kdisk==None and args.shuffle:
#             if args.shuffle=='sh':
#                 pass