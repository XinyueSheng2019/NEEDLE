# from random import shuffle
import pandas as pd 
import numpy as np 
import h5py
import json
import os
import datetime
import random

def open_with_h5py(filepath):
    imageset = np.array(h5py.File(filepath, mode = 'r')['imageset'])
    labels = np.array(h5py.File(filepath, mode = 'r')['label'])
    metaset = np.array(h5py.File(filepath, mode = 'r')['metaset'])
    idx_set = np.array(h5py.File(filepath, mode = 'r')['idx_set'])
    return imageset, labels, metaset, idx_set

def single_transient_preprocessing(image, meta, band, add_host):
    image, meta = np.array(image), np.array(meta)
    pre_image = image.reshape(1, image.shape[0], image.shape[1], image.shape[-1])
    pre_meta = meta.reshape(1, meta.shape[0])
    return pre_image, pre_meta


def data_scaling(metaset, output_path, normalize_method = 1):
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

        s_data = {'mean': mt_mean.astype('float64').tolist(), 'std': mt_std.astype('float64').tolist()}
    elif normalize_method == 'normal_by_sample' or normalize_method == 2:
        # metaset normalization
        mt_min = np.nanmin(metaset, axis = 1)[:,np.newaxis]
        mt_max = np.nanmax(metaset, axis = 1)[:,np.newaxis]
        b_metaset, b_min, b_max = np.broadcast_arrays(metaset, mt_min, mt_max)
        metaset = (b_metaset-b_min)/(b_max - b_min)
        s_data = {}
    elif normalize_method == 'both' or normalize_method == 3:
        # standarlise by feature
        mt_mean = np.nanmean(metaset, axis=0)
        mt_std = np.nanstd(metaset, axis=0)
        norf_metaset = (metaset - mt_mean)/mt_std
        s_data = {'mean': mt_mean.astype('float64').tolist(), 'std': mt_std.astype('float64').tolist()}

        # metaset normalization
        mt_min = np.nanmin(metaset, axis = 1)[:,np.newaxis]
        mt_max = np.nanmax(metaset, axis = 1)[:,np.newaxis]
        b_metaset, b_min, b_max = np.broadcast_arrays(metaset, mt_min, mt_max)
        nors_metaset = (b_metaset-b_min)/(b_max - b_min)
        metaset = np.concatenate((norf_metaset, nors_metaset), axis = -1)

    with open(output_path + '/scaling_data.json', 'w') as sd:
            json.dump(s_data, sd, indent = 4)

    return metaset 

def apply_data_scaling(metaset, scaling_file, normalize_method = 1):

    if isinstance(scaling_file, str):
        f = open(scaling_file, 'r')
        scaling = json.load(f)
        f.close()
    else:
        scaling = scaling_file

    if normalize_method == 'normal_by_feature' or normalize_method == 0:
        metaset = (metaset - np.array(scaling['min']))/(np.array(scaling['max']) - np.array(scaling['min']))
    
    elif normalize_method == 'standarlize_by_feature' or normalize_method == 1:
        # standarlise by feature
        metaset = (metaset - np.array(scaling['mean']))/np.array(scaling['std'])

    elif normalize_method == 'normal_by_sample' or normalize_method == 2:
        # metaset normalization

        b_metaset, b_min, b_max = np.broadcast_arrays(metaset, mt_min, mt_max)
        metaset = (b_metaset-b_min)/(b_max - b_min)

    elif normalize_method == 'both' or normalize_method == 3:
        # standarlise by feature
        norf_metaset = (metaset - np.array(scaling['mean']))/np.array(scaling['std'])
       
        # metaset normalization
        mt_min = np.nanmin(metaset, axis = 1)[:,np.newaxis]
        mt_max = np.nanmax(metaset, axis = 1)[:,np.newaxis]
        b_metaset, b_min, b_max = np.broadcast_arrays(metaset, mt_min, mt_max)
        nors_metaset = (b_metaset-b_min)/(b_max - b_min)

        metaset = np.concatenate((norf_metaset, nors_metaset), axis = -1)

    return metaset 


def feature_reduction_for_mixed_band(metadata):
    # mixed_nor1_add_disc_t_ext_20240628
    print(metadata.shape)

    feature_names = ['candi_mag_r', 'disc_mag_r', 'delta_mag_discovery_r', 'delta_t_discovery_band_r', 'delta_t_discovery_r', 'ratio_recent_r', 'ratio_disc_r', 'delta_host_mag_r',
                 'candi_mag_g', 'disc_mag_g', 'delta_mag_discovery_g', 'delta_t_discovery_band_g', 'delta_t_discovery_g', 'ratio_recent_g', 'ratio_disc_g', 'delta_host_mag_g',
                  'peak_mag_g_minus_r', 'peak_t_g_minus_r', 
                  'host_g','host_r','host_i','host_z','host_y', 'host_g-r', 'host_r-i', 
                  'offset']
    df = pd.DataFrame(metadata, columns=feature_names)
    df['host_i-z'] = df['host_i'] - df['host_z']
    df['host_z-y'] = df['host_z'] - df['host_y']
    df['ratio_dff_r']  = df['ratio_disc_r'] - df['ratio_recent_r']
    df['ratio_dff_g']  = df['ratio_disc_g'] - df['ratio_recent_g']
    df['disc_mag_g_minus_r'] = df.apply(lambda row: 0 if row['disc_mag_g'] == 0 or row['disc_mag_r'] == 0 else row['disc_mag_g'] - row['disc_mag_r'], axis=1)
    df['colour_dff'] = df.apply(lambda row: 0 if row['peak_mag_g_minus_r'] == 0 or row['disc_mag_g_minus_r'] == 0 else row['peak_mag_g_minus_r'] - row['disc_mag_g_minus_r'], axis=1)
    df['host_tar_colour_g-r'] = df['delta_host_mag_g'] - df['delta_host_mag_r']
    df = df.drop(['ratio_recent_r', 'ratio_recent_g', 'delta_t_discovery_band_r', 'delta_t_discovery_band_g'], axis = 1)
    return df.to_numpy(), df.columns

def feature_reduction_for_mixed_band_no_host(metadata):
    feature_names = ['candi_mag_r', 'disc_mag_r', 'delta_mag_discovery_r', 'delta_t_discovery_band_r', 'delta_t_discovery_r', 'ratio_recent_r', 'ratio_disc_r',
                 'candi_mag_g', 'disc_mag_g', 'delta_mag_discovery_g', 'delta_t_discovery_band_g', 'delta_t_discovery_g', 'ratio_recent_g', 'ratio_disc_g', 
                  'peak_mag_g_minus_r', 'peak_t_g_minus_r']
    df = pd.DataFrame(metadata, columns=feature_names)
    df['ratio_dff_r']  = df['ratio_disc_r'] - df['ratio_recent_r']
    df['ratio_dff_g']  = df['ratio_disc_g'] - df['ratio_recent_g']
    df['disc_mag_g_minus_r'] = df.apply(lambda row: 0 if row['disc_mag_g'] == 0 or row['disc_mag_r'] == 0 else row['disc_mag_g'] - row['disc_mag_r'], axis=1)
    df['colour_dff'] = df.apply(lambda row: 0 if row['peak_mag_g_minus_r'] == 0 or row['disc_mag_g_minus_r'] == 0 else row['peak_mag_g_minus_r'] - row['disc_mag_g_minus_r'], axis=1)
    df = df.drop(['ratio_recent_r', 'ratio_recent_g', 'delta_t_discovery_band_r', 'delta_t_discovery_band_g'], axis = 1)
    return df.to_numpy(), df.columns


def get_feature_ranking(X_train, y_train, class_weight, feature_names, model_path = None, feature_ranking_path = None):
    if feature_ranking_path is None:
        if model_path is not None:
            import xgboost as xgb
            sample_weights = np.array([class_weight[cls] for cls in y_train])
            best_xgb_params = {'subsample': 0.8, 'n_estimators': 200, 'min_child_weight': 2, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0.1, 'colsample_bytree': 0.6}
            xgb_model= xgb.XGBClassifier(**best_xgb_params)
            xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
        return xgb_model.feature_importances_
    else:
        f = open(feature_ranking_path, 'r')
        data = json.load(f)
        f.close()
        return [data[x] for x in feature_names]

    


def get_class_weight(labels):
    class_weight = {}
    for i in np.arange(len(set(labels.flatten()))):
        class_weight[i] = labels.shape[0]/len(np.where(labels.flatten()==i)[0])
    return class_weight

def preprocessing(filepath, label_dict, hash_path, output_path, normalize_method = 1, custom_path = None, band = None, feature_ranking_path = None, add_host = True):

    imageset, labels, metaset, idx_set = open_with_h5py(filepath)

    hash_table = open(hash_path, 'r')
    hash_table = json.loads(hash_table.read())

    # reverse hash table
    reversed_hash = {}
    for i in hash_table.keys():
        reversed_hash[hash_table[i]['ztf_id']] = int(i)
   

    # reverse hash labels
    reversed_label = {}
    for i in hash_table.keys():
        if hash_table[i]['label'] not in reversed_label.keys():
            reversed_label[hash_table[i]['label']] = []

        reversed_label[hash_table[i]['label']].append(int(i))
    

    test_num_dict = label_dict["test_num"]

    for k in label_dict['classify'].keys():
        if label_dict['classify'][k] not in label_dict['label'].values():
            ab_idx = np.where(labels == label_dict["classify"][k])
            imageset, metaset, labels = np.delete(imageset, ab_idx, 0), np.delete(metaset, ab_idx, 0), np.delete(labels, ab_idx, 0)
            idx_set = np.delete(idx_set, ab_idx, 0)

    if not os.path.exists(output_path):
            os.makedirs(output_path)
  
    # seperate training and test sets from hash table
    if custom_path is None:
        test_obj_dict = {}
        test_idx = []
        for k in test_num_dict.keys():
            obj_index = np.where(labels == label_dict["label"][k])
            obj_idx = idx_set[obj_index]
            np.random.seed()
            np.random.shuffle(obj_idx)
            test_k_idx = obj_idx[:test_num_dict[k]]
            k_idx_set = np.nonzero(np.isin(idx_set, test_k_idx))[0].tolist()
            test_idx += k_idx_set
            test_obj_dict[k] = {}
            for j in test_k_idx:
                test_obj_dict[k][hash_table[str(int(j))]["ztf_id"]] = str(int(j))
        
        # write test_obj_dict.json
        with open(output_path + "/testset_obj.json", "w") as outfile:
            json.dump(test_obj_dict, outfile, indent = 4)
    else:
        test_obj_dict = {}
        custom_obj = json.load(open(custom_path, 'r'))
        test_idx = []
        for k in label_dict['label'].keys():
            if k in custom_obj.keys():
                k_id = list(custom_obj[k])
                test_k_idx = []
                for ki in k_id:
                    test_k_idx.append(reversed_hash[ki])
            else:
                # shuffle SNe samples for every model
                obj_idx = reversed_label[label_dict['label'][k]]
                np.random.seed()
                np.random.shuffle(obj_idx)
                test_k_idx = obj_idx[:15]

            # use the test idx, to find their indices in whole set.
            k_idx_set = np.nonzero(np.isin(idx_set, test_k_idx))[0].tolist()
            test_idx += k_idx_set
            
            # record the test idx into a JSON file.
            test_obj_dict[k] = {}
            for j in test_k_idx:
                test_obj_dict[k][hash_table[str(int(j))]["ztf_id"]] = str(int(j))
            
        with open(output_path + "/testset_obj.json", "w") as outfile:
            json.dump(test_obj_dict, outfile, indent = 4)

    train_imageset, train_metaset, train_labels = np.delete(imageset, test_idx, 0), np.delete(metaset, test_idx, 0), np.delete(labels, test_idx, 0)
    test_imageset, test_metaset, test_labels = np.take(imageset, test_idx, 0), np.take(metaset, test_idx, 0), np.take(labels, test_idx, 0)

    train_imageset = np.nan_to_num(train_imageset)
    train_metaset = np.nan_to_num(train_metaset)
    test_imageset = np.nan_to_num(test_imageset)
    test_metaset = np.nan_to_num(test_metaset)

    # print('preprocess: ',train_imageset.shape, train_metaset.shape, test_imageset.shape)
    if band == 'mixed':
        if add_host:
            train_metaset, feature_names = feature_reduction_for_mixed_band(train_metaset)
            test_metaset, _ = feature_reduction_for_mixed_band(test_metaset)
        else:
            train_metaset, feature_names = feature_reduction_for_mixed_band_no_host(train_metaset)
            test_metaset, _ = feature_reduction_for_mixed_band_no_host(test_metaset)
        class_weight = get_class_weight(train_labels)
        feature_importances = get_feature_ranking(train_metaset, train_labels, class_weight, feature_names, output_path, feature_ranking_path)
        train_metaset = data_scaling(train_metaset, output_path, normalize_method)
        test_metaset = apply_data_scaling(test_metaset, output_path + '/scaling_data.json', normalize_method)
        return train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels, feature_importances
    else:
        train_metaset = data_scaling(train_metaset, output_path, normalize_method)
        test_metaset = apply_data_scaling(test_metaset, output_path + '/scaling_data.json', normalize_method)
        return train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels, None


def select_customised_objs(train_validation_list, reversed_hash):
    '''
    combine a training and a validation set with customized SLSN-I or TDE sets.
    '''
    train_validation_set = {}
    for v in train_validation_list:
        train_validation_set[v] = reversed_hash[v]
    
    return train_validation_set



def custom_preprocessing(filepath, label_dict, hash_path, output_path, normalize_method = 1, custom_path = None, object_with_host_path = None):
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



# if __name__ == '__main__':

#     label_path = 'label_dict.json'
#     label_dict = open(label_path,'r')
#     label_dict = json.loads(label_dict.read())

#     hash_path = 'hash_table.json'
#     # hash_table = open(hash_path, 'r')
#     # hash_table = json.loads(hash_table.read())

#     filepath = 'r_peak_set.hdf5'

#     preprocessing(filepath, label_dict, hash_path)
    

     