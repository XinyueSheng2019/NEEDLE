#! ~/anaconda3/envs/astro_py8/bin/python python3
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
from build_dataset import single_band_peak_db, gr_band_peak_db
from preprocessing import preprocessing, cut_preprocessing, single_transient_preprocessing,open_with_h5py
from training import train
from tensorflow.keras import models
from ztf_image_pipeline import collect_image, read_table
from sherlock_host_pipeline import get_potential_host
from host_meta_pipeline import PS1catalog_host
from obj_meta_pipeline import collect_meta
from build_dataset import get_single_transient_peak
import config
import argparse


def build_and_train_models(band, image_path, host_path, mag_path, output_path, label_path, quality_model_path, no_diff = True, only_complete = True, add_host = False, neurons = [[128,5],[128,5]], res_cnn_group = None, batch_size = 32, epoch = 300, learning_rate = 5e-5, model_name = None, custom_test_path = None):
   
    label_dict = open(label_path,'r')
    label_dict = json.loads(label_dict.read())
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        BClassifier = models.load_model(quality_model_path) 
        if band != 'gr':
            single_band_peak_db(image_path, host_path, mag_path, output_path, label_dict["classify"], band = band, no_diff= no_diff, only_complete = only_complete, add_host = add_host, BClassifier = BClassifier)
        else:
            gr_band_peak_db(image_path, host_path, mag_path, output_path, label_dict['classify'], no_diff = no_diff, add_host = add_host, only_complete = only_complete, BClassifier = BClassifier)
    else:
        print('Data already exist! Start training.\n')
    filepath = output_path + 'data.hdf5'
    hash_path = output_path + 'hash_table.json'
   
    model_path = output_path + model_name

    train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels = preprocessing(filepath, label_dict, hash_path, model_path, 1, custom_test_path)
    # train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels = cut_preprocessing(filepath, label_dict, hash_path, model_path, 1, custom_test_path, object_with_host_path = '/Users/xinyuesheng/Documents/astro_projects/scripts/classifier_v2/model_with_data/r_band/v13_v2_3c_20231126/hash_table.json')
    print(train_imageset.shape, train_metaset.shape)
    
    train(train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels, label_dict["label"], neurons = neurons, res_cnn_group = res_cnn_group, batch_size = batch_size, epoch = epoch, learning_rate = learning_rate, model_name = model_path)


def add_single_transient(ztf_id, disdate, transient_type, size, duration, outdir, magdir, hostdir):

    print('---------collect %s image data now---------\n'%(ztf_id))
    try:
        collect_image(ztf_id, disdate, transient_type, size, duration, outdir, magdir)
    except ValueError:
        print('The image download process appears an error.\n')


    f = open(outdir + '/'+ ztf_id + '/image_meta.json')
    meta = json.load(f)
    f.close()

    print('---------get top host coordinates from Sherlock.---------\n')
    host_ra, host_dec = get_potential_host(meta['ra'], meta['dec'])
    if host_ra is None or host_dec is None:
        print('---------WARNING! host no found.---------\n')
    else:
        print('---------HOST FOUND: ra = %f dec = %f---------\n'%(host_ra, host_dec))

    print('---------get host meta from PanSTARR---------\n')
    PS1catalog_host(ztf_id, host_ra, host_dec, radius = 0.0014, save_path = hostdir)

    print('---------get object meta---------\n')
    collect_meta(ztf_id, outdir, hostdir)

    print('---------%s is added successfully!---------\n'%(ztf_id))

    


def add_multiple_transients(transient_table, size, duration, outdir, magdir, parrallel = True):

    read_table(transient_table, size, duration, outdir, magdir, parrallel = parrallel)

    for ztf_id in transient_table['ztf_id'].tolist():
        f = open(outdir + '/'+ ztf_id + '/image_meta.json')
        meta = json.load(f)
        f.close()
        print('---------collect %s host and obj metadata now.---------\n'%(ztf_id))
        print('---------get top host coordinates from Sherlock.---------\n')
        host_ra, host_dec = get_potential_host(meta['ra'], meta['dec'])
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
    # print(img_data, meta_data)
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




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='NEEDLE_training')
    parser.add_argument("-i", help="iteration with Makefile (Ignore)")
    args = vars(parser.parse_args())

    build_and_train_models(config.BAND, config.IMAGE_PATH, config.HOST_PATH, config.MAG_PATH, config.OUTPUT_PATH, config.LABEL_PATH, config.QUALITY_MODEL_PATH, 
                               config.NO_DIFF, config.ONLY_COMPLETE, config.ADD_HOST, config.NEURONS,  config.RES_CNN_GROUP,  
                               config.BATCH_SIZE, config.EPOCH, config.LEARNING_RATE, 
                               model_name='seed_' + str(config.SEED) + config.MODEL_NAME + args['i'])
    
   
