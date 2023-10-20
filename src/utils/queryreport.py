import logging
import os
import csv
import numpy as np

from tqdm import tqdm

from utils.utils import event_duration, user_response_exe, path, query_report, query_yes_no, query_yes_no_exe, rename_path
from datagen.datagent import datagen, datagenHdf5, create_kdtree, create_kdtree_slab_shuffled, create_kdtree_slab_matrix_shuffled
from dataloader.dataloader import dataload
from crypto import permute

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)") 
ed = event_duration(scale='milli', log=False)

def report_hist(args, indx=''):
    user_response = False

    path_inst = path()
    path_inst = rename_path(path_inst, indx)
    ds_size = np.prod(args.datadim[1:])

    report_dict = {
               'kdtree':{'en_no_disk':[], 'en_sh_disk':[], 'en_shs_disk':[], 'en_shsm_disk':[]}}

    i_start = tuple(args.rangequery[0::2])
    i_finish = tuple(args.rangequery[1::2])
    dataLoader = dataload(path_inst.org_data,
                        data_dim=args.datadim[0],
                        lat_dim=args.datadim[1],
                        long_dim=args.datadim[2],
                        height_dim=args.datadim[3],
                        time_dim=args.datadim[4],
                        encrypted=None)
    args.query, args.encrypt, args.shuffle = 'txt', None, None
    queryResult, _ = query_report(args, ed, [], dataLoader.load_slabs, i_start, i_finish)

    # query=='kdtree' and encrypt==True and shuffle==None and kdisk==True
    dataLoader = dataload(path_inst.org_encrypted,
                        data_dim=args.datadim[0],
                        lat_dim=args.datadim[1],
                        long_dim=args.datadim[2],
                        height_dim=args.datadim[3],
                        time_dim=args.datadim[4],
                        encrypted = None)
    args.query, args.encrypt, args.shuffle = 'kdtree', True, None
    _, report_dict['kdtree']['en_no_disk'] = query_report(args, ed, queryResult[0], dataLoader.range_query_kdtree, 
                            i_start, i_finish, path_inst.kdtree_encrypted, 'disk')
    report_dict['kdtree']['en_no_disk'] = add_info(report_dict['kdtree']['en_no_disk'], path_inst.kdtree_encrypted,
                                                    args.datadim, dataLoader.buckets_ctr, part_nums=dataLoader.part_nums)
 
    # query=='kdtree' and encrypt==True and shuffle=='sh' and kdisk==True
    dataLoader = dataload(path_inst.encrypted_shuffled,
                        data_dim=args.datadim[0],
                        lat_dim=args.datadim[1],
                        long_dim=args.datadim[2],
                        height_dim=args.datadim[3],
                        time_dim=args.datadim[4],
                        encrypted = None,
                        shuffled='element',
                        map_dir=path_inst.mapping)
    args.query, args.encrypt, args.shuffle = 'kdtree', True, 'sh'
    _, report_dict['kdtree']['en_sh_disk'] = query_report(args, ed, queryResult[0], dataLoader.range_query_kdtree, 
                        i_start, i_finish, path_inst.kdtree_encrypted_shuffled, 'disk')
    report_dict['kdtree']['en_sh_disk'] = add_info(report_dict['kdtree']['en_sh_disk'], path_inst.kdtree_encrypted_shuffled, args.datadim,
                                                    dataLoader.buckets_ctr, part_nums=dataLoader.part_nums)

    # query=='kdtree' and encrypt==True and shuffle=='shs' and kdisk==True
    dataLoader = dataload(path_inst.encrypted_shuffled_slab,
                        data_dim=args.datadim[0],
                        lat_dim=args.datadim[1],
                        long_dim=args.datadim[2],
                        height_dim=args.datadim[3],
                        time_dim=args.datadim[4],
                        encrypted = None,
                        shuffled='slab',
                        map_dir=path_inst.mapping_slab)
    args.query, args.encrypt, args.shuffle = 'kdtree', True, 'shs'
    _, report_dict['kdtree']['en_shs_disk'] = query_report(args, ed, queryResult[0], dataLoader.range_query_kdtree_slab, 
                        i_start, i_finish, path_inst.kdtree_encrypted_shuffled_slab, args.slab_size)
    report_dict['kdtree']['en_shs_disk'] = add_info(report_dict['kdtree']['en_shs_disk'], path_inst.kdtree_encrypted_shuffled_slab,
                                                     args.datadim, dataLoader.buckets_ctr, part_nums=dataLoader.part_nums)
    
    # query=='kdtree' and encrypt==True and shuffle=='shsm' and kdisk==True
    dataLoader = dataload(path_inst.encrypted_shuffled_slab_matrix,
                    data_dim=args.datadim[0],
                    lat_dim=args.datadim[1],
                    long_dim=args.datadim[2],
                    height_dim=args.datadim[3],
                    time_dim=args.datadim[4],
                    encrypted = None,
                    shuffled='slab_matrix',
                    map_dir=path_inst.mapping_slab_matrix)
    # range_query_kdtree_slab_matrix(self, r_start, r_finish, kdtree_filePath, slab_dim, data_dim)
    args.query, args.encrypt, args.shuffle = 'kdtree', True, 'shsm'
    _, report_dict['kdtree']['en_shsm_disk'] = query_report(args, ed, queryResult[0], dataLoader.range_query_kdtree_slab_matrix, 
                        i_start, i_finish, path_inst.kdtree_encrypted_shuffled_slab_matrix, args.slabdim, args.datadim)
    report_dict['kdtree']['en_shsm_disk'] = add_info(report_dict['kdtree']['en_shsm_disk'], path_inst.kdtree_encrypted_shuffled_slab_matrix, 
                                                     args.datadim, dataLoader.buckets_ctr, part_nums = dataLoader.part_nums)

    write_report_2(report_dict, path_inst.report_bucket, args)

def add_info(list_info, file_path, datadim, backet_ctr = None, part_nums = 0):
    added_data = [os.path.getsize(file_path), datadim[1], datadim[2], datadim[3], datadim[4], np.prod(datadim[1:]), backet_ctr, part_nums]
    for i in added_data:
        list_info.append(i)
    return list_info

def write_report(report_dict, file_path, header):
    values = []
    for k,v in report_dict.items():
        for k2,v2 in v.items():
            row = [k+'_'+k2,*v2[:-1]]
            values.append(row)

    if not os.path.exists(file_path):
        with open(file_path, 'w'):
                pass
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(values)

    # logging.info(f"The report is written in the {file_path}")

def write_report_2(report_dict, file_path, args):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['method_name', 'bucket_count', 'query size','response_size', 'bucket_count/response_size','part numbers',
                             'width1','width2','width3', 'width4','start1', 'finish1','start2', 'finish2','start3', 'finish3','start4', 'finish4']) 

    for k,v in report_dict.items():
        for k2,v2 in v.items():
            width = [e-i for i,e in zip(args.rangequery[0::2], args.rangequery[1::2])]
            #final repoted data in the row
            query_title = [k+'_'+k2, len(v2[-2].keys()), np.prod(width), sum(v2[-2].values()),
                            round(len(v2[-2].keys())/sum(v2[-2].values()),2),v2[-1], *width, *args.rangequery]
                
            with open(file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(query_title)

    # logging.info(f"The report is written in the {file_path}")