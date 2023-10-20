import logging
import os
import csv
import numpy as np

from utils.utils import event_duration, user_response_exe, path, query_report, query_yes_no, query_yes_no_exe, rename_path
from datagen.datagent import datagen, create_kdtree, create_kdtree_slab_shuffled, create_kdtree_slab_matrix_shuffled
from dataloader.dataloader import dataload
from crypto import permute

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)") 
ed = event_duration(scale='milli', log=True)

def report(args, prompt = True, indx=''):
    if prompt == True:
        user_response = query_yes_no(f"If the file already exists, do you want to overwrite it?")
    else:
        user_response = True

    path_inst = path()
    path_inst = rename_path(path_inst, indx)
    ds_size = np.prod(args.datadim[1:])
    # --------------------- txt ---------------------
    dataGenerator_org = datagen(path_inst.org_data,
                            data_dim=args.datadim[0],
                            lat_dim=args.datadim[1],
                            long_dim=args.datadim[2],
                            height_dim=args.datadim[3],
                            time_dim=args.datadim[4])
    # datagen=='txt' and encrypt==None and shuffle==None
    user_response_exe(path_inst.org_data,  user_response, dataGenerator_org.generate_data)
    # datagen=='txt' and encrypt==True and shuffle==None
    user_response_exe(path_inst.org_encrypted, user_response, 
                        dataGenerator_org.generate_encryptData_from_txt, path_inst.org_encrypted, 'AES_XTS')
    # datagen=='txt' and encrypt==True and shuffle=='sh'
    ds_pos=list(range(2,ds_size+2))
    user_response_exe(path_inst.mapping, user_response, permute.prf, ds_pos, 
                        path_inst.mapping, args.datadim, path_inst.line_2_ijkm)
    user_response_exe(path_inst.encrypted_shuffled, user_response, permute.permute, 
                        path_inst.org_encrypted, path_inst.encrypted_shuffled, path_inst.mapping)
    # datagen=='txt' and encrypt==True and shuffle=='sh0'
    user_response_exe(path_inst.mapping_index0, user_response, permute.prf_block, path_inst.mapping_index0, 
                        args.datadim, 1)
    user_response_exe(path_inst.encrypted_shuffled_index0, user_response, permute.permute_indx_0, 
                        path_inst.org_encrypted, path_inst.encrypted_shuffled_index0, 
                        path_inst.mapping_index0, args.datadim)

    # datagen=='txt' and encrypt==True and shuffle=='shs'
    user_response_exe(path_inst.mapping_slab, user_response, permute.prf_slab, args.slab_size, ds_size, path_inst.mapping_slab)
    user_response_exe(path_inst.encrypted_shuffled_slab, user_response, permute.permute_slab,
                        path_inst.org_encrypted, path_inst.encrypted_shuffled_slab, 
                        path_inst.mapping_slab, args.slab_size)
    
    # datagen=='txt' and encrypt==True and shuffle=='shsm'
    slab_set_dim = tuple([int(args.datadim[i+1]/args.slabdim[i]) for i in range(len(args.datadim[1:]))])
    org_data = list(range(1,np.prod(slab_set_dim)+1))
    user_response_exe(path_inst.mapping_slab_matrix, user_response, permute.prf_general, org_data, path_inst.mapping_slab_matrix)
    user_response_exe(path_inst.encrypted_shuffled_slab_matrix, user_response, permute.permute_slab_matrix,
                        path_inst.org_encrypted, path_inst.encrypted_shuffled_slab_matrix, 
                        path_inst.mapping_slab_matrix, args.slabdim, args.datadim)    

    # --------------------- kdtree ---------------------

    # datagen=='kdtree' and encrypt==True and shuffle==None and kdisk==True
    user_response_exe(path_inst.kdtree_encrypted, user_response, create_kdtree, args.datadim, 
                        path_inst.org_encrypted, path_inst.kdtree_encrypted, 'disk')
    # datagen=='kdtree' and encrypt==True and shuffle==None and kdisk==None
    user_response_exe(path_inst.kdtree_encrypted_obj, user_response, create_kdtree, args.datadim, 
                        path_inst.org_encrypted, path_inst.kdtree_encrypted_obj, 'memory')
    # datagen=='kdtree' and encrypt==True and shuffle=='sh' and kdisk==True
    user_response_exe(path_inst.kdtree_encrypted_shuffled, user_response, create_kdtree, args.datadim, 
                        path_inst.encrypted_shuffled, path_inst.kdtree_encrypted_shuffled, 'disk')
    # datagen=='kdtree' and encrypt==True and shuffle=='sh' and kdisk==None
    user_response_exe(path_inst.kdtree_encrypted_shuffled_obj, user_response, create_kdtree, 
                        args.datadim, path_inst.encrypted_shuffled, 
                        path_inst.kdtree_encrypted_shuffled_obj, 'memory')
    # datagen=='kdtree' and encrypt==True and shuffle=='shs' and kdisk==True
    user_response_exe(path_inst.kdtree_encrypted_shuffled_slab, user_response, create_kdtree_slab_shuffled,
                       args.datadim, path_inst.encrypted_shuffled_slab, path_inst.kdtree_encrypted_shuffled_slab, args.slab_size)
    # datagen=='kdtree' and encrypt==True and shuffle=='shsm' and kdisk==True
    user_response_exe(path_inst.kdtree_encrypted_shuffled_slab_matrix, user_response, create_kdtree_slab_matrix_shuffled, args.datadim, 
                        path_inst.encrypted_shuffled_slab_matrix, path_inst.kdtree_encrypted_shuffled_slab_matrix, args.slabdim)    

    report_dict = {'txt':{'no_no':[],'en_no':[],  'en_sh':[], 'en_sh0':[]},
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

    # query=='txt' and encrypt==None and shuffle==None
    args.query, args.encrypt, args.shuffle = 'txt', None, None
    _, report_dict['txt']['no_no']  = query_report(args, ed, queryResult[0], dataLoader.load_slabs, i_start, i_finish)
    report_dict['txt']['no_no'] = add_info(report_dict['txt']['no_no'], path_inst.org_data, args.datadim, dataLoader.buckets_ctr)
    # query=='txt' and encrypt==True and shuffle==None
    dataLoader = dataload(path_inst.org_encrypted,
                data_dim=args.datadim[0],
                lat_dim=args.datadim[1],
                long_dim=args.datadim[2],
                height_dim=args.datadim[3],
                time_dim=args.datadim[4],
                encrypted='AES_XTS')
    args.query, args.encrypt, args.shuffle = 'txt', True, None
    _, report_dict['txt']['en_no'] = query_report(args, ed, queryResult[0], dataLoader.load_slabs, i_start, i_finish)
    report_dict['txt']['en_no'] = add_info(report_dict['txt']['en_no'], path_inst.org_encrypted, args.datadim, dataLoader.buckets_ctr)
    # query=='txt' and encrypt==True and shuffle=='sh'
    dataLoader = dataload(path_inst.encrypted_shuffled,
            data_dim=args.datadim[0],
            lat_dim=args.datadim[1],
            long_dim=args.datadim[2],
            height_dim=args.datadim[3],
            time_dim=args.datadim[4],
            encrypted='AES_XTS',
            shuffled='element',
            map_dir=path_inst.mapping)
    args.query, args.encrypt, args.shuffle = 'txt', True, 'sh'
    _, report_dict['txt']['en_sh'] = query_report(args, ed, queryResult[0], dataLoader.load_slabs, i_start, i_finish)
    report_dict['txt']['en_sh'] = add_info(report_dict['txt']['en_sh'], path_inst.encrypted_shuffled, args.datadim, dataLoader.buckets_ctr)
    # query=='txt' and encrypt==True and shuffle=='sh0'
    dataLoader = dataload(path_inst.encrypted_shuffled_index0,
            data_dim=args.datadim[0],
            lat_dim=args.datadim[1],
            long_dim=args.datadim[2],
            height_dim=args.datadim[3],
            time_dim=args.datadim[4],
            encrypted='AES_XTS',
            shuffled='index_0',
            map_dir=path_inst.mapping_index0,
            line_2_ijkm_dir=path_inst.line_2_ijkm)
    args.query, args.encrypt, args.shuffle = 'txt', True, 'sh0'
    _, report_dict['txt']['en_sh0'] = query_report(args, ed, queryResult[0], dataLoader.load_slabs, i_start, i_finish, path_inst.mapping_index0)
    report_dict['txt']['en_sh0'] = add_info(report_dict['txt']['en_sh0'], path_inst.encrypted_shuffled_index0, args.datadim, dataLoader.buckets_ctr)

    # query=='kdtree' and encrypt==True and shuffle==None and kdisk==True
    dataLoader = dataload(path_inst.org_encrypted,
                        data_dim=args.datadim[0],
                        lat_dim=args.datadim[1],
                        long_dim=args.datadim[2],
                        height_dim=args.datadim[3],
                        time_dim=args.datadim[4],
                        encrypted = 'AES_XTS')
    args.query, args.encrypt, args.shuffle = 'kdtree', True, None
    _, report_dict['kdtree']['en_no_disk'] = query_report(args, ed, queryResult[0], dataLoader.range_query_kdtree, 
                            i_start, i_finish, path_inst.kdtree_encrypted, 'disk')
    report_dict['kdtree']['en_no_disk'] = add_info(report_dict['kdtree']['en_no_disk'], path_inst.kdtree_encrypted, args.datadim, dataLoader.buckets_ctr)

    # query=='kdtree' and encrypt==True and shuffle=='sh' and kdisk==True
    dataLoader = dataload(path_inst.encrypted_shuffled,
                        data_dim=args.datadim[0],
                        lat_dim=args.datadim[1],
                        long_dim=args.datadim[2],
                        height_dim=args.datadim[3],
                        time_dim=args.datadim[4],
                        encrypted = 'AES_XTS',
                        shuffled='element',
                        map_dir=path_inst.mapping)
    args.query, args.encrypt, args.shuffle = 'kdtree', True, 'sh'
    _, report_dict['kdtree']['en_sh_disk'] = query_report(args, ed, queryResult[0], dataLoader.range_query_kdtree, 
                        i_start, i_finish, path_inst.kdtree_encrypted_shuffled, 'disk')
    report_dict['kdtree']['en_sh_disk'] = add_info(report_dict['kdtree']['en_sh_disk'], path_inst.kdtree_encrypted_shuffled, args.datadim, dataLoader.buckets_ctr)

    # query=='kdtree' and encrypt==True and shuffle=='shs' and kdisk==True
    dataLoader = dataload(path_inst.encrypted_shuffled_slab,
                        data_dim=args.datadim[0],
                        lat_dim=args.datadim[1],
                        long_dim=args.datadim[2],
                        height_dim=args.datadim[3],
                        time_dim=args.datadim[4],
                        encrypted = 'AES_XTS',
                        shuffled='slab',
                        map_dir=path_inst.mapping_slab)
    args.query, args.encrypt, args.shuffle = 'kdtree', True, 'shs'
    _, report_dict['kdtree']['en_shs_disk'] = query_report(args, ed, queryResult[0], dataLoader.range_query_kdtree_slab, 
                        i_start, i_finish, path_inst.kdtree_encrypted_shuffled_slab, args.slab_size)
    report_dict['kdtree']['en_shs_disk'] = add_info(report_dict['kdtree']['en_shs_disk'], path_inst.kdtree_encrypted_shuffled_slab, args.datadim, dataLoader.buckets_ctr)
    
    # query=='kdtree' and encrypt==True and shuffle=='shsm' and kdisk==True
    dataLoader = dataload(path_inst.encrypted_shuffled_slab_matrix,
                    data_dim=args.datadim[0],
                    lat_dim=args.datadim[1],
                    long_dim=args.datadim[2],
                    height_dim=args.datadim[3],
                    time_dim=args.datadim[4],
                    encrypted = 'AES_XTS',
                    shuffled='slab_matrix',
                    map_dir=path_inst.mapping_slab_matrix)
    # range_query_kdtree_slab_matrix(self, r_start, r_finish, kdtree_filePath, slab_dim, data_dim)
    args.query, args.encrypt, args.shuffle = 'kdtree', True, 'shsm'
    _, report_dict['kdtree']['en_shsm_disk'] = query_report(args, ed, queryResult[0], dataLoader.range_query_kdtree_slab_matrix, 
                        i_start, i_finish, path_inst.kdtree_encrypted_shuffled_slab_matrix, args.slabdim, args.datadim)
    report_dict['kdtree']['en_shsm_disk'] = add_info(report_dict['kdtree']['en_shsm_disk'], path_inst.kdtree_encrypted_shuffled_slab_matrix, args.datadim, dataLoader.buckets_ctr)

    header = ['query', 'time', 'valid', 'query_size', 'file_size', 'dim1', 'dim2', 'dim3', 'dim4', 'data_set_size']
    # bucket_headers = ['Bucket_Number', 'Number_of_Accesses']

    if prompt == True:    
        query_yes_no_exe(path_inst.report, write_report, report_dict, path_inst.report, header)
        query_yes_no_exe(path_inst.report_bucket, write_report_2, report_dict, path_inst.report_bucket)
    else:
        write_report(report_dict, path_inst.report, header)
        write_report_2(report_dict, path_inst.report_bucket)

def add_info(list_info, file_path, datadim, backet_ctr = None):
    added_data = [os.path.getsize(file_path), datadim[1], datadim[2], datadim[3], datadim[4], np.prod(datadim[1:]), backet_ctr]
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

    logging.info(f"The report is written in the {file_path}")

def write_report_2(report_dict, file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['bucket_num', 'count']) 

    for k,v in report_dict.items():
        for k2,v2 in v.items():
            values = []
            if len(v2)!=0:
                for k3,v3 in v2[-1].items():
                    values.append([k3,v3])
                
            with open(file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                query_title = [k+'_'+k2, None, len(values), sum([i[1] for i in values])]
                writer.writerow(query_title)
                writer.writerows(values)

    logging.info(f"The report is written in the {file_path}")