import numpy as np
import h5py
import json
import os
import datetime


def open_with_h5py(filepath):
    with h5py.File(filepath, mode='r') as file:
        imageset = np.array(file['imageset'])
        labels = np.array(file['label'])
        metaset = np.array(file['metaset'])
        idx_set = np.array(file['idx_set'])
    return imageset, labels, metaset, idx_set


def single_transient_preprocessing(image, meta):
    image, meta = np.array(image), np.array(meta)
    pre_image = image.reshape(1, image.shape[0], image.shape[1], image.shape[-1])
    pre_meta = meta.reshape(1, meta.shape[0])
    return pre_image, pre_meta


def data_scaling(metaset, output_path, normalize_method=1):
    s_data = {}
    if normalize_method in ['normal_by_feature', 0]:
        # Normalize by feature
        mt_min = np.nanmin(metaset, axis=0)
        mt_max = np.nanmax(metaset, axis=0)
        metaset = (metaset - mt_min) / (mt_max - mt_min)
        s_data = {'max': mt_max.astype('float64').tolist(), 'min': mt_min.astype('float64').tolist()}
    elif normalize_method in ['standarlize_by_feature', 1]:
        # Standardize by feature
        mt_mean = np.nanmean(metaset, axis=0)
        mt_std = np.nanstd(metaset, axis=0)
        metaset = (metaset - mt_mean) / mt_std
        s_data = {'mean': mt_mean.astype('float64').tolist(), 'std': mt_std.astype('float64').tolist()}
    elif normalize_method in ['normal_by_sample', 2]:
        # Normalize by sample
        mt_min = np.nanmin(metaset, axis=1)[:, np.newaxis]
        mt_max = np.nanmax(metaset, axis=1)[:, np.newaxis]
        metaset = (metaset - mt_min) / (mt_max - mt_min)
    elif normalize_method in ['both', 3]:
        # Standardize by feature
        mt_mean = np.nanmean(metaset, axis=0)
        mt_std = np.nanstd(metaset, axis=0)
        norf_metaset = (metaset - mt_mean) / mt_std
        s_data = {'mean': mt_mean.astype('float64').tolist(), 'std': mt_std.astype('float64').tolist()}
        # Normalize by sample
        mt_min = np.nanmin(metaset, axis=1)[:, np.newaxis]
        mt_max = np.nanmax(metaset, axis=1)[:, np.newaxis]
        nors_metaset = (metaset - mt_min) / (mt_max - mt_min)
        metaset = np.concatenate((norf_metaset, nors_metaset), axis=-1)

    with open(os.path.join(output_path, 'scaling_data.json'), 'w') as sd:
        json.dump(s_data, sd, indent=4)

    return metaset


def preprocessing(filepath, label_dict, hash_path, output_path, normalize_method=1, custom_path=None):
    imageset, labels, metaset, idx_set = open_with_h5py(filepath)

    with open(hash_path, 'r') as f:
        hash_table = json.load(f)

    # Reverse hash table
    reversed_hash = {hash_table[i]['ztf_id']: int(i) for i in hash_table}
    # Reverse hash label
    reversed_label = {}
    for i in hash_table:
        label = hash_table[i]['label']
        if label not in reversed_label:
            reversed_label[label] = []
        reversed_label[label].append(int(i))

    test_num_dict = label_dict["test_num"]

    # Filter out unwanted classes
    for k, v in label_dict['classify'].items():
        if v not in label_dict['label'].values():
            ab_idx = np.where(labels == v)
            imageset, metaset, labels = np.delete(imageset, ab_idx, 0), np.delete(metaset, ab_idx, 0), np.delete(labels, ab_idx, 0)
            idx_set = np.delete(idx_set, ab_idx, 0)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    metaset = data_scaling(metaset, output_path, normalize_method)

    # Split training and test sets
    if custom_path is None:
        test_obj_dict = {}
        test_idx = []
        for k, v in test_num_dict.items():
            obj_index = np.where(labels == label_dict["label"][k])
            obj_idx = idx_set[obj_index]
            np.random.seed(datetime.datetime.now().second * (datetime.datetime.now().minute + 1))
            np.random.shuffle(obj_idx)
            test_k_idx = obj_idx[:v]
            k_idx_set = np.nonzero(np.isin(idx_set, test_k_idx))[0].tolist()
            test_idx += k_idx_set
            test_obj_dict[k] = {hash_table[str(int(j))]["ztf_id"]: str(int(j)) for j in test_k_idx}

        with open(os.path.join(output_path, "testset_obj.json"), "w") as outfile:
            json.dump(test_obj_dict, outfile, indent=4)
    else:
        test_obj_dict = {}
        with open(custom_path, 'r') as f:
            custom_obj = json.load(f)
        test_idx = []
        for k, v in label_dict['label'].items():
            if k in custom_obj:
                test_k_idx = [reversed_hash[ki] for ki in custom_obj[k]]
            else:
                obj_idx = reversed_label[v]
                np.random.seed(datetime.datetime.now().second * (datetime.datetime.now().minute + 1))
                np.random.shuffle(obj_idx)
                test_k_idx = obj_idx[:50]
            k_idx_set = np.nonzero(np.isin(idx_set, test_k_idx))[0].tolist()
            test_idx += k_idx_set
            test_obj_dict[k] = {hash_table[str(int(j))]["ztf_id"]: str(int(j)) for j in test_k_idx}

        with open(os.path.join(output_path, "testset_obj.json"), "w") as outfile:
            json.dump(test_obj_dict, outfile, indent=4)

    train_imageset, train_metaset, train_labels = np.delete(imageset, test_idx, 0), np.delete(metaset, test_idx, 0), np.delete(labels, test_idx, 0)
    test_imageset, test_metaset, test_labels = np.take(imageset, test_idx, 0), np.take(metaset, test_idx, 0), np.take(labels, test_idx, 0)

    train_imageset = np.nan_to_num(train_imageset)
    train_metaset = np.nan_to_num(train_metaset)
    test_imageset = np.nan_to_num(test_imageset)
    test_metaset = np.nan_to_num(test_metaset)

    # Return the processed datasets
    return train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels


def select_customised_objs(train_validation_list, reversed_hash):
    '''
    Combine a training and a validation set with customized SLSN-I or TDE sets.
    '''
    train_validation_set = {v: reversed_hash[v] for v in train_validation_list}
    return train_validation_set


def custom_preprocessing(filepath, label_dict, hash_path, output_path, normalize_method=1, custom_path=None, object_with_host_path=None):
    '''
    Pick training objects by custom selection.
    '''
    imageset, labels, metaset, idx_set = open_with_h5py(filepath)
    with open(hash_path, 'r') as t:
        hash_table = json.load(t)
    
    test_num_dict = label_dict["test_num"]

    # Filter out unwanted classes
    for k, v in label_dict['classify'].items():
        if v not in label_dict['label'].values():
            ab_idx = np.where(labels == v)
            imageset, metaset, labels = np.delete(imageset, ab_idx, 0), np.delete(metaset, ab_idx, 0), np.delete(labels, ab_idx, 0)
            idx_set = np.delete(idx_set, ab_idx, 0)

    # Select objects with hosts only
    with open(object_with_host_path, 'r') as f:
        with_host_hash = json.load(f)

    with_host_objs = [with_host_hash[i]["ztf_id"] for i in with_host_hash]
    
    reversed_hash = {hash_table[i]["ztf_id"]: int(i) for i in hash_table}

    mag_host_index = [np.where(idx_set == reversed_hash[i])[0][0] for i in with_host_objs if i in reversed_hash]

    imageset, metaset, labels, idx_set = imageset[mag_host_index], metaset[mag_host_index], labels[mag_host_index], idx_set[mag_host_index]
    
    metaset = data_scaling(metaset, output_path, normalize_method)

    # Separate training and test sets
    if custom_path is None:
        test_obj_dict = {}
        test_idx = []
        for k, v in test_num_dict.items():
            obj_idx = np.where(labels == label_dict["label"][k])[0]
            np.random.seed(datetime.datetime.now().second * (datetime.datetime.now().minute + 1))
            np.random.shuffle(obj_idx)
            test_k_idx = obj_idx[:v]
            test_idx += test_k_idx.tolist()
            k_idx_set = idx_set[test_k_idx]
            test_obj_dict[k] = {hash_table[str(int(j))]["ztf_id"]: str(int(j)) for j in k_idx_set}

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, "testset_obj.json"), "w") as outfile:
            json.dump(test_obj_dict, outfile, indent=4)
    else:
        with open(custom_path, 'r') as f:
            testset_obj = json.load(f)
        test_idx = [int(testset_obj[k][ztf_id]) for k in label_dict['label'] for ztf_id in testset_obj[k]]
        sorter = idx_set.argsort()
        test_idx = sorter[np.searchsorted(idx_set, test_idx, sorter=sorter)]
    
    train_imageset, train_metaset, train_labels = np.delete(imageset, test_idx, 0), np.delete(metaset, test_idx, 0), np.delete(labels, test_idx, 0)
    test_imageset, test_metaset, test_labels = np.take(imageset, test_idx, 0), np.take(metaset, test_idx, 0), np.take(labels, test_idx, 0)

    train_imageset = np.nan_to_num(train_imageset)
    train_metaset = np.nan_to_num(train_metaset)
    test_imageset = np.nan_to_num(test_imageset)
    test_metaset = np.nan_to_num(test_metaset)

    return train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels


def test_file():
    label_path = 'label_dict_equal_test.json'
    with open(label_path, 'r') as f:
        label_dict = json.load(f)
    
    hash_path = '/Users/xinyuesheng/Documents/astro_projects/scripts/classifier_v2/model_with_data/r_band/test_build_dataset/hash_table.json'
    filepath = '/Users/xinyuesheng/Documents/astro_projects/scripts/classifier_v2/model_with_data/r_band/test_build_dataset/data.hdf5'
    output_path = '/Users/xinyuesheng/Documents/astro_projects/scripts/classifier_v2/model_with_data/r_band/test_build_dataset/'  # Define the output path
    
    train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels = preprocessing(
        filepath, label_dict, hash_path, output_path, normalize_method=0
    )
    
    # Print shapes to verify output
    print('Train Imageset Shape:', train_imageset.shape)
    print('Train Metaset Shape:', train_metaset.shape)
    print('Train Labels Shape:', train_labels.shape)
    print('Test Imageset Shape:', test_imageset.shape)
    print('Test Metaset Shape:', test_metaset.shape)
    print('Test Labels Shape:', test_labels.shape)



# test_file() # PASS

