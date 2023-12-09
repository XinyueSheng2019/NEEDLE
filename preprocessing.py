# from random import shuffle
import pandas as pd 
import numpy as np 
from astropy import visualization
import h5py
import json
import os
import tensorflow as tf
import datetime


def open_with_h5py(filepath):
    imageset = np.array(h5py.File(filepath, mode = 'r')['imageset'])
    labels = np.array(h5py.File(filepath, mode = 'r')['label'])
    metaset = np.array(h5py.File(filepath, mode = 'r')['metaset'])
    idx_set = np.array(h5py.File(filepath, mode = 'r')['idx_set'])
    return imageset, labels, metaset, idx_set

def single_transient_preprocessing(image, meta):
    image, meta = np.array(image), np.array(meta)
    pre_image = image.reshape(1, image.shape[0], image.shape[1], image.shape[-1])
    pre_meta = meta.reshape(1,meta.shape[0])
    return pre_image, pre_meta

def preprocessing(filepath, label_dict, hash_path, output_path, normalize_method = 1, custom_path = None):

    imageset, labels, metaset, idx_set = open_with_h5py(filepath)
    hash_table = open(hash_path, 'r')
    hash_table = json.loads(hash_table.read())

  
    
    test_num_dict = label_dict["test_num"]

    for k in label_dict['classify'].keys():
        if label_dict['classify'][k] not in label_dict['label'].values():
            ab_idx = np.where(labels == label_dict["classify"][k])
            imageset, metaset, labels = np.delete(imageset, ab_idx, 0), np.delete(metaset, ab_idx, 0), np.delete(labels, ab_idx, 0)
            idx_set = np.delete(idx_set, ab_idx, 0)
    
    if normalize_method == 'normal_by_feature' or normalize_method == 0:
        # normalize by feature
        mt_min = np.nanmin(metaset, axis = 0)
        mt_max = np.nanmax(metaset, axis = 0)
        metaset = (metaset - mt_min)/(mt_max - mt_min)
        s_data = {'max': mt_max.astype('float64').tolist(), 'min': mt_min.astype('float64').tolist()}
    elif normalize_method == 'standarlize_by_feature' or normalize_method == 1:
        # standarlise by feature
        mt_mean = np.nanmean(metaset, axis=0)
        mt_std = np.nanstd(metaset, axis=0)
        metaset = (metaset - mt_mean)/mt_std
        print(mt_std)
        print(mt_mean)
        s_data = {'mean': mt_mean.astype('float64').tolist(), 'std': mt_std.astype('float64').tolist()}
    elif normalize_method == 'normal_by_sample' or normalize_method == 2:
        # metaset normalization
        mt_min = np.nanmin(metaset, axis = 1)[:,np.newaxis]
        mt_max = np.nanmax(metaset, axis = 1)[:,np.newaxis]
        b_metaset, b_min, b_max = np.broadcast_arrays(metaset, mt_min, mt_max)
        metaset = (b_metaset-b_min)/(b_max - b_min)
        s_data = {}

    # seperate training and test sets from hash table
    if custom_path is None:
        test_obj_dict = {}
        test_idx = []
        for k in test_num_dict.keys():
            obj_idx = np.where(labels == label_dict["label"][k])[0]
            # set a random seed based on the time!
            np.random.seed(datetime.datetime.now().second * (datetime.datetime.now().minute+1))
            np.random.shuffle(obj_idx)
            test_k_idx = obj_idx[:test_num_dict[k]]
            test_idx += test_k_idx.tolist()
            k_idx_set = idx_set[test_k_idx]
            test_obj_dict[k] = {}
            for j in k_idx_set:
                test_obj_dict[k][hash_table[str(int(j))]["ztf_id"]] = str(int(j))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # write test_obj_dict.json
        with open(output_path + "/testset_obj.json", "w") as outfile:
            json.dump(test_obj_dict, outfile, indent = 4)
        with open(output_path + '/scaling_data.json', 'w') as sd:
            json.dump(s_data, sd, indent = 4)
    else:
        testset_obj = json.load(open(custom_path, 'r'))
        test_idx = []
        for k in label_dict['label'].keys():
            test_idx += testset_obj[k].values()
        test_idx = np.array(test_idx).astype('int32')
        sorter = idx_set.argsort()
        test_idx = np.array(sorter[np.searchsorted(idx_set, test_idx, sorter=sorter)])

    train_imageset, train_metaset, train_labels = np.delete(imageset, test_idx, 0), np.delete(metaset, test_idx, 0), np.delete(labels, test_idx, 0)
    test_imageset, test_metaset, test_labels = np.take(imageset, test_idx, 0), np.take(metaset, test_idx, 0), np.take(labels, test_idx, 0)

    train_imageset = np.nan_to_num(train_imageset)
    train_metaset = np.nan_to_num(train_metaset)
    test_imageset = np.nan_to_num(test_imageset)
    test_metaset = np.nan_to_num(test_metaset)

    return train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels

def cut_preprocessing(filepath, label_dict, hash_path, output_path, normalize_method = 1, custom_path = None, object_with_host_path = None):
    '''
    pick training ojects by custom.
    '''
    imageset, labels, metaset, idx_set = open_with_h5py(filepath)
    t = open(hash_path, 'r')
    hash_table = json.loads(t.read())
    t.close()
  
    test_num_dict = label_dict["test_num"]

    for k in label_dict['classify'].keys():
        if label_dict['classify'][k] not in label_dict['label'].values():
            ab_idx = np.where(labels == label_dict["classify"][k])
            imageset, metaset, labels = np.delete(imageset, ab_idx, 0), np.delete(metaset, ab_idx, 0), np.delete(labels, ab_idx, 0)
            idx_set = np.delete(idx_set, ab_idx, 0)
    
    # select objects with hosts only
    f = open(object_with_host_path, 'r')
    with_host_hash = json.loads(f.read())
    f.close()

    with_host_objs = []
    for i in with_host_hash.keys(): 
        with_host_objs.append(with_host_hash[i]["ztf_id"])
     
    reversed_hash = {}
    for i in hash_table.keys():
        reversed_hash[hash_table[i]["ztf_id"]] = int(i)

    mag_host_index = []
    for i in with_host_objs:
        if i in reversed_hash.keys():
            arg_idx = np.where(idx_set==reversed_hash[i])[0]
            if len(arg_idx) == 1:
                mag_host_index.append(arg_idx[0])

    mag_host_index = np.array(mag_host_index).astype('int32')

    print(len(mag_host_index), len(set(mag_host_index)))

    imageset, metaset, labels, idx_set = imageset[mag_host_index], metaset[mag_host_index], labels[mag_host_index], idx_set[mag_host_index]
        

    if normalize_method == 'normal_by_feature' or normalize_method == 0:
        # normalize by feature
        mt_min = np.nanmin(metaset, axis = 0)
        mt_max = np.nanmax(metaset, axis = 0)
        metaset = (metaset - mt_min)/(mt_max - mt_min)
        s_data = {'max': mt_max.astype('float64').tolist(), 'min': mt_min.astype('float64').tolist()}
    elif normalize_method == 'standarlize_by_feature' or normalize_method == 1:
        # standarlise by feature
        mt_mean = np.nanmean(metaset, axis=0)
        mt_std = np.nanstd(metaset, axis=0)
        metaset = (metaset - mt_mean)/mt_std
        # print(mt_std)
        # print(mt_mean)
        s_data = {'mean': mt_mean.astype('float64').tolist(), 'std': mt_std.astype('float64').tolist()}
    elif normalize_method == 'normal_by_sample' or normalize_method == 2:
        # metaset normalization
        mt_min = np.nanmin(metaset, axis = 1)[:,np.newaxis]
        mt_max = np.nanmax(metaset, axis = 1)[:,np.newaxis]
        b_metaset, b_min, b_max = np.broadcast_arrays(metaset, mt_min, mt_max)
        metaset = (b_metaset-b_min)/(b_max - b_min)
        s_data = {}

    # seperate training and test sets from hash table
    if custom_path is None:
        test_obj_dict = {}
        test_idx = []
        for k in test_num_dict.keys():
            obj_idx = np.where(labels == label_dict["label"][k])[0]
            # set a random seed based on the time!
            np.random.seed(datetime.datetime.now().second * (datetime.datetime.now().minute+1))
            np.random.shuffle(obj_idx)
            test_k_idx = obj_idx[:test_num_dict[k]]
            test_idx += test_k_idx.tolist()
            k_idx_set = idx_set[test_k_idx]
            test_obj_dict[k] = {}
            for j in k_idx_set:
                test_obj_dict[k][hash_table[str(int(j))]["ztf_id"]] = str(int(j))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # write test_obj_dict.json
        with open(output_path + "/testset_obj.json", "w") as outfile:
            json.dump(test_obj_dict, outfile, indent = 4)
        with open(output_path + '/scaling_data.json', 'w') as sd:
            json.dump(s_data, sd, indent = 4)
        
    else:
        testset_obj = json.load(open(custom_path, 'r'))
        test_idx = []
        for k in label_dict['label'].keys():
            test_idx += testset_obj[k].values()
        test_idx = np.array(test_idx).astype('int32')
        sorter = idx_set.argsort()
        test_idx = np.array(sorter[np.searchsorted(idx_set, test_idx, sorter=sorter)])
        with open(output_path + '/scaling_data.json', 'w') as sd:
            json.dump(s_data, sd, indent = 4)

    train_imageset, train_metaset, train_labels = np.delete(imageset, test_idx, 0), np.delete(metaset, test_idx, 0), np.delete(labels, test_idx, 0)
    test_imageset, test_metaset, test_labels = np.take(imageset, test_idx, 0), np.take(metaset, test_idx, 0), np.take(labels, test_idx, 0)

    train_imageset = np.nan_to_num(train_imageset)
    train_metaset = np.nan_to_num(train_metaset)
    test_imageset = np.nan_to_num(test_imageset)
    test_metaset = np.nan_to_num(test_metaset)

    return train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels



if __name__ == '__main__':

    label_path = 'label_dict.json'
    label_dict = open(label_path,'r')
    label_dict = json.loads(label_dict.read())

    hash_path = 'hash_table.json'
    # hash_table = open(hash_path, 'r')
    # hash_table = json.loads(hash_table.read())

    filepath = 'r_peak_set.hdf5'

    preprocessing(filepath, label_dict, hash_path)
    

        # print(np.all(np.isnan(imageset)))

    # image normalization: in the build_dataset process
    # print(imageset[1,1,:].shape)
    # imageset = np.apply_along_axis(zscale, 2, imageset)
    # imageset = np.apply_along_axis(image_normalization, 2, imageset)

    # meta normailization: we have the NormalizationLayer in the model, so this step is removed.

    # re-label based on requirements
    # for k in relabel_dict["classify"].keys():
    #     for j in relabel_dict["classify"][k]:
    #         labels[labels == j+100] = relabel_dict["relabel"][k]
    