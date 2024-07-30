
'''
User interaction
achieve below functions:
1. add new objects to train, validation and test sets
2. train a model with user-defined parameters
3. get the Confusion Matrix plots
4. get the intepretation plots
'''
import numpy as np
import os, json
from build_dataset import single_band_peak_db, both_band_peak_db, mixed_band_peak_db
from preprocessing import preprocessing, custom_preprocessing, single_transient_preprocessing, open_with_h5py, apply_data_scaling, feature_reduction_for_mixed_band, feature_reduction_for_mixed_band_no_host
from training import train
from tensorflow.keras import models
from ztf_image_pipeline import collect_image_from_irsa, read_table
from sherlock_host_pipeline import get_potential_host_from_json
from host_meta_pipeline import PS1catalog_host
from obj_meta_pipeline import collect_meta
from build_dataset import get_single_transient_peak
import pandas as pd 
import config
import argparse


def build_and_train_models(band, image_path, host_path, mag_path, output_path, label_path, quality_model_path, no_diff = True, only_complete = True, add_host = False, meta_only = False, neurons = [[128,5],[128,5]], res_cnn_group = None, batch_size = 32, epoch = 300, learning_rate = 5e-5, model_name = None, custom_test_path = None, object_with_host_path = None, normalize_method = 1, note = None, feature_ranking_path = None):
   
    label_dict = open(label_path,'r')
    label_dict = json.loads(label_dict.read())
    
    filepath = output_path + 'data.hdf5'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(filepath):
        BClassifier = None
        if quality_model_path is not None:
            BClassifier = models.load_model(quality_model_path) 
        if band == 'g' or band == 'r':
            single_band_peak_db(image_path, host_path, mag_path, output_path, label_dict["classify"], band = band, no_diff= no_diff, only_complete = only_complete, add_host = add_host, BClassifier = BClassifier)
        elif band == 'gr':
            both_band_peak_db(image_path, host_path, mag_path, output_path, label_dict['classify'], no_diff = no_diff, add_host = add_host, only_complete = only_complete, BClassifier = BClassifier)
        elif band == 'mixed':
            mixed_band_peak_db(image_path, host_path, mag_path, output_path, label_dict['classify'], no_diff = no_diff, add_host = add_host, only_complete = only_complete, BClassifier = BClassifier)
        else:
            print('INVALID BAND!')

    else:
        print('Data already exist! Start training.\n')
    
    hash_path = output_path + 'hash_table.json'
    model_path = output_path + model_name

    if object_with_host_path is None:
        train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels, feature_importance = preprocessing(filepath, label_dict, hash_path, model_path, normalize_method, custom_test_path, band, feature_ranking_path, add_host) 
    else:
        train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels = custom_preprocessing(filepath, label_dict, hash_path, model_path, normalize_method, custom_test_path, object_with_host_path)
    
    print(train_imageset.shape, train_metaset.shape, test_imageset.shape)
    
    train(train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels, feature_importance, label_dict["label"], neurons = neurons, res_cnn_group = res_cnn_group, meta_only = meta_only, 
          batch_size = batch_size, epoch = epoch, learning_rate = learning_rate, model_name = model_path, note = note )



def build_dataset(band, image_path, host_path, mag_path, output_path, label_path, quality_model_path, no_diff = True, only_complete = True, add_host = False):
    '''
    Build a dataset for training or test.
    '''
    label_dict = open(label_path,'r')
    label_dict = json.loads(label_dict.read())

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    BClassifier = models.load_model(quality_model_path) 

    if band == 'g' or band == 'r':
        single_band_peak_db(image_path, host_path, mag_path, output_path, label_dict["classify"], band = band, no_diff = no_diff, only_complete = only_complete, add_host = add_host, BClassifier = BClassifier)
    elif band == 'gr':
        both_band_peak_db(image_path, host_path, mag_path, output_path, label_dict['classify'], no_diff = no_diff, add_host = add_host, only_complete = only_complete, BClassifier = BClassifier)
    elif band == 'mixed':
        mixed_band_peak_db(image_path, host_path, mag_path, output_path, label_dict['classify'], no_diff = no_diff, add_host = add_host, only_complete = only_complete, BClassifier = BClassifier)
    else:
        print('INVALID BAND!')


def add_single_transient(ztf_id, disdate, transient_type, size, duration, outdir, magdir, hostdir):

    print('---------collect %s image data now---------\n'%(ztf_id))
    try:
        collect_image_from_irsa(ztf_id, disdate, transient_type, size, duration, outdir, magdir)
    except ValueError:
        print('The image download process appears an error.\n')


    f = open(outdir + '/'+ ztf_id + '/image_meta.json')
    meta = json.load(f)
    f.close()

    print('---------get top host coordinates from Sherlock.---------\n')
    host_ra, host_dec = get_potential_host_from_json(meta['ra'], meta['dec'])
    if host_ra is None or host_dec is None:
        print('---------WARNING! host no found.---------\n')
    else:
        print('---------HOST FOUND: ra = %f dec = %f---------\n'%(host_ra, host_dec))

    print('---------get host meta from PanSTARR---------\n')
    PS1catalog_host(ztf_id, host_ra, host_dec, radius = 0.0014, save_path = hostdir)

    print('---------get object meta---------\n')
    collect_meta(ztf_id, outdir, hostdir)

    print('---------%s is added successfully!---------\n'%(ztf_id))

    


def add_multiple_transients(transient_table, size, duration, outdir, magdir, hostdir, parrallel = True):

    read_table(transient_table, size, duration, outdir, magdir, parrallel = parrallel)

    for ztf_id in transient_table['object_id']:
        f = open(outdir + '/'+ ztf_id + '/image_meta.json')
        meta = json.load(f)
        f.close()
        print('---------collect %s host and obj metadata now.---------\n'%(ztf_id))
        print('---------get top host coordinates from Sherlock.---------\n')
        host_ra, host_dec = get_potential_host_from_json(ztf_id, magdir)

        if host_ra is None or host_dec is None:
            print('---------WARNING! host no found.---------\n')
        else:
            print('---------HOST FOUND: ra = %f dec = %f---------\n'%(host_ra, host_dec))

        print('---------get host meta from PanSTARR---------\n')
        PS1catalog_host(ztf_id, host_ra, host_dec, radius = 0.0014, save_path = hostdir)

        print('---------get object meta---------\n')
        collect_meta(ztf_id, outdir, hostdir)

        print('---------%s is added successfully!---------\n'%(ztf_id))


def predict_new_transient(ztf_id, disdate, label_path, BClassifier_path, TSClassifier_path, predict_path = 'new_predicts/'):
   
    if os.path.isdir(predict_path) == False:
        os.mkdir(predict_path)
    magdir = predict_path + 'mags/'
    if os.path.isdir(magdir) == False:
        os.mkdir(magdir)
    outdir = predict_path + 'images/'
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)
    hostdir = predict_path + 'hosts/'
    if os.path.isdir(hostdir) == False:
        os.mkdir(hostdir)
    f = open(label_path)
    label_dict = json.load(f)['label']
    f.close()

    TSClassifier = models.load_model(TSClassifier_path + '/model')

    BClassifier = models.load_model(BClassifier_path)

    transient_type = 'unknown'
    size = 1
    duration = 50
    
    add_single_transient(ztf_id, disdate, transient_type, size, duration, outdir, magdir, hostdir)
    img_data, meta_data = get_single_transient_peak(ztf_id, outdir, hostdir, band = 'r', no_diff = True, BClassifier = BClassifier)
    # print(img_data.shape, meta_data.shape)
    img_data, meta_data = single_transient_preprocessing(img_data, meta_data)

    results = TSClassifier.predict({'image_input': img_data, 'meta_input': meta_data})

    print('Prediction: %s:%f, %s:%f, %s:%f\n' % (list(label_dict.keys())[0], results[0][0], list(label_dict.keys())[1], results[0][1], list(label_dict.keys())[2], results[0][2]))


def predict_test_transient(ztf_id, transient_type, label_path, TSClassifier_path, predict_path = 'test_predicts'):
 
    TSClassifier = models.load_model(TSClassifier_path + '/model')


    t = open(TSClassifier_path + '/testset_obj.json')
    testset_obj = json.load(t)
    t.close()

    f = open(label_path)
    label_dict = json.load(f)['label']
    f.close()

    if ztf_id not in testset_obj[transient_type]:
        raise ValueError('ZTF ID IS NOT FOUND IN TEST SET. TRY FUNCTION predict_new_transient().\n')
    else:
        idx = testset_obj[transient_type][ztf_id]
        imageset, labels, metaset, idx_set = open_with_h5py(TSClassifier_path + '/data.hdf5')
        obj_index = np.where(idx_set == int(idx))
        img, meta = imageset[obj_index], metaset[obj_index]
        results = TSClassifier.predict({'image_input': img, 'meta_input': meta})
        print('Prediction: %s:%f, %s:%f, %s:%f\n' % (list(label_dict.keys())[0], results[0][0], list(label_dict.keys())[1], results[0][1], list(label_dict.keys())[2], results[0][2]))

def scaling_meta(meta_data, scaling_file_path):
    '''
    assume normaliztion method 1 in this case.
    '''
    f = open(scaling_file_path+'/scaling_data.json')
    scaling = json.load(f)
    f.close()
    mt_mean = np.array(scaling['mean'])
    mt_std = np.array(scaling['std'])
    meta_data = (meta_data - mt_mean)/mt_std
    return meta_data

def predict_new_dataset(data_path, needle_path, threshold, output = None, normalize_method = 3, mixed = False, add_host = True):
    LABEL_LIST = ['SN', 'SLSN-I', 'TDE']
    img_data, _, meta_data, idx_set = open_with_h5py(data_path + '/data.hdf5')
    f = open(data_path + '/hash_table.json')
    hash_table = json.load(f)
    f.close()

    img_data = np.nan_to_num(img_data)
    meta_data = np.nan_to_num(meta_data)


    emsemble_results = []
    for i in np.arange(5):
        TSClassifier = models.load_model(needle_path + str(i))
        scaling_file_path = needle_path + str(i) + '/scaling_data.json'
        
        if mixed:
            if add_host:
                _meta_data, _= feature_reduction_for_mixed_band(meta_data) 
            else:
                _meta_data, _= feature_reduction_for_mixed_band_no_host(meta_data) 

        else:
            _meta_data = meta_data
        _meta_data = apply_data_scaling(_meta_data, scaling_file_path, normalize_method)

        results = TSClassifier.predict({'image_input': img_data, 'meta_input': _meta_data})
        emsemble_results.append(results)

    emsemble_results = np.array(emsemble_results)
    final_results = np.mean(emsemble_results, axis = 0)

    # output into a csv file

    result_data = []
    n = 0
    for i in hash_table.keys():
        result = final_results[np.where(idx_set==int(i))[0][0]]
        probs = 'SN: %f, SLSN-I: %f, TDE: %f' % (result[0], result[1], result[2])
        if np.max(result) >= threshold:
            classification = LABEL_LIST[np.argmax(result)]
        else:
            classification = 'unclear'
        max_class = LABEL_LIST[np.argmax(result)]
        result_data.append([hash_table[i]['ztf_id'], hash_table[i]['type'], classification, max_class, probs])
        n += 1
    
    df = pd.DataFrame(result_data, columns = ['ztf_id', 'type', f'p>={threshold}', 'highest score', 'probability'])
    df = df.sort_values(by=['type'])
    print(df.to_string())
    

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NEEDLE_training')
    parser.add_argument("-i", help="iteration with Makefile (Ignore)")
    args = vars(parser.parse_args())

    build_and_train_models(config.BAND, config.IMAGE_PATH, config.HOST_PATH, config.MAG_PATH, config.OUTPUT_PATH, config.LABEL_PATH, config.QUALITY_MODEL_PATH, 
                               config.NO_DIFF, config.ONLY_COMPLETE, config.ADD_HOST, config.META_ONLY,config.NEURONS,  config.RES_CNN_GROUP,
                               config.BATCH_SIZE, config.EPOCH, config.LEARNING_RATE, 
                               model_name='seed_' + str(config.SEED) + config.MODEL_NAME + args['i'], custom_test_path = config.CUSTOM_TEST_PATH, object_with_host_path = config.OBJECT_WITH_HOST_PATH, normalize_method=config.META_NORMALIZE, note = config.NOTE, feature_ranking_path = config.FEATURE_RANKING_PATH)
    



    # band = 'mixed'
    # image_path = '/Users/xinyuesheng/Documents/astro_projects/data/untouch_testset/images'
    # host_path = '/Users/xinyuesheng/Documents/astro_projects/data/untouch_testset/hosts_ext'
    # mag_path = '/Users/xinyuesheng/Documents/astro_projects/data/untouch_testset/mags'
    # output_path = f'/Users/xinyuesheng/Documents/astro_projects/data/untouch_testset/{band}_band'
    # label_path = '/Users/xinyuesheng/Documents/astro_projects/scripts/classifier_v1/model_labels/label_dict_equal_test.json'
    # quality_model_path = '../../quality_classifier/models/bogus_model_without_zscale'
    # needle_path = f'/Users/xinyuesheng/Documents/astro_projects/scripts/classifier_v1/model_with_data/{band}_band/mixed_nor12_20240531/seed_456_model_nor1_neurons_32_128_256_'
    # threshold = 0.75
    # # build_dataset(band, image_path, host_path, mag_path, output_path, label_path, quality_model_path, no_diff = True, only_complete = True, add_host = True)
    # predict_new_dataset(output_path, needle_path, threshold,f'{band}_nor1', 1)