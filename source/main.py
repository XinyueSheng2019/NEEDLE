'''
User interaction
achieve below functions:
1. add new objects to train, validation and test sets
2. train a model with user-defined parameters
3. get the Confusion Matrix plots
4. get the intepretation plots
'''

import numpy as np
import os
import json
import argparse
from tensorflow.keras import models
from build_dataset import single_band_peak_db, both_band_peak_db, get_single_transient_peak
from preprocessing import preprocessing, single_transient_preprocessing, open_with_h5py
from training import train
from ztf_image_pipeline import collect_image
from sherlock_host_pipeline import get_potential_host
from host_meta_pipeline import PS1catalog_host
from obj_meta_pipeline import collect_meta
from transient_model import EM_QualityClassifier
import sys
sys.path.append('../')
import config as config

def build_and_train_models(band, image_path, host_path, mag_path, output_path, label_path, quality_model_path,
                            no_diff=True, only_complete=True, add_host=False, neurons=[[128, 5], [128, 5]], 
                            res_cnn_group=None, batch_size=32, epoch=300, learning_rate=5e-5, model_name=None, 
                            custom_test_path=None):
    label_dict = json.load(open(label_path, 'r'))
    if not os.path.exists(os.path.join(output_path, 'data.hdf5')):
        os.makedirs(output_path, exist_ok=True)
        BClassifier = EM_QualityClassifier(model_path=quality_model_path)

        if band != 'gr':
            single_band_peak_db(image_path, host_path, mag_path, output_path, label_dict["classify"], band=band,
                                no_diff=no_diff, only_complete=only_complete, add_host=add_host,
                                BClassifier=BClassifier)
        else:
            both_band_peak_db(image_path, host_path, mag_path, output_path, label_dict['classify'], no_diff=no_diff,
                            add_host=add_host, only_complete=only_complete, BClassifier=BClassifier)
    else:
        print('Data already exists! Start training.\n')

    filepath = os.path.join(output_path, 'data.hdf5')
    hash_path = os.path.join(output_path, 'hash_table.json')
    model_path = os.path.join(output_path, model_name)

    train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels = preprocessing(
        filepath, label_dict, hash_path, model_path, 1, custom_test_path)

    print(train_imageset.shape, train_metaset.shape)

    train(train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels, label_dict["label"],
          neurons=neurons, batch_size=batch_size, epochs=epoch,
          learning_rate=learning_rate, model_name=model_path)


def add_single_transient(ztf_id, disdate, transient_type, size, duration, outdir, magdir, hostdir):
    print(f'---------collect {ztf_id} image data now---------\n')
    try:
        collect_image(ztf_id, disdate, transient_type, size, duration, outdir, magdir)
    except ValueError:
        print('The image download process encountered an error.\n')

    with open(os.path.join(outdir, ztf_id, 'image_meta.json')) as f:
        meta = json.load(f)

    print('---------get top host coordinates from Sherlock.---------\n')
    host_ra, host_dec = get_potential_host(meta['ra'], meta['dec'])
    if host_ra is None or host_dec is None:
        print('---------WARNING! host not found.---------\n')
    else:
        print('---------HOST FOUND: ra = %f dec = %f---------\n' % (host_ra, host_dec))

    print('---------get host meta from PanSTARR---------\n')
    PS1catalog_host(ztf_id, host_ra, host_dec, radius=0.0014, save_path=hostdir)

    print('---------get object meta---------\n')
    collect_meta(ztf_id, outdir, hostdir)

    print(f'---------{ztf_id} is added successfully!---------\n')

def predict_new_transient(ztf_id, disdate, label_path, BClassifier_path, TSClassifier_path, predict_path='new_predicts/'):
    if not os.path.isdir(predict_path):
        os.makedirs(predict_path)

    magdir = os.path.join(predict_path, 'mags/')
    outdir = os.path.join(predict_path, 'images/')
    hostdir = os.path.join(predict_path, 'hosts/')

    for directory in [magdir, outdir, hostdir]:
        if not os.path.isdir(directory):
            os.makedirs(directory)

    with open(label_path) as f:
        label_dict = json.load(f)['label']

    TSClassifier = models.load_model(os.path.join(TSClassifier_path, 'model'))
    BClassifier = models.load_model(BClassifier_path)

    transient_type = 'unknown'
    size = 1
    duration = 50

    add_single_transient(ztf_id, disdate, transient_type, size, duration, outdir, magdir, hostdir)
    img_data, meta_data = get_single_transient_peak(ztf_id, outdir, hostdir, band='r', no_diff=True,
                                                     BClassifier=BClassifier)
    img_data, meta_data = single_transient_preprocessing(img_data, meta_data)
    results = TSClassifier.predict({'image_input': img_data, 'meta_input': meta_data})

    print('Prediction: %s:%f, %s:%f, %s:%f\n' % (
    list(label_dict.keys())[0], results[0][0], list(label_dict.keys())[1], results[0][1], list(label_dict.keys())[2],
    results[0][2]))

def predict_test_transient(ztf_id, transient_type, label_path, TSClassifier_path, predict_path='test_predicts'):
    TSClassifier = models.load_model(os.path.join(TSClassifier_path, 'model'))
    with open(TSClassifier_path + '/testset_obj.json') as t:
        testset_obj = json.load(t)

    with open(label_path) as f:
        label_dict = json.load(f)['label']

    if ztf_id not in testset_obj[transient_type]:
        raise ValueError('ZTF ID IS NOT FOUND IN TEST SET. TRY FUNCTION predict_new_transient().\n')

    idx = testset_obj[transient_type][ztf_id]
    imageset, labels, metaset, idx_set = open_with_h5py(os.path.join(TSClassifier_path, 'data.hdf5'))
    obj_index = np.where(idx_set == int(idx))
    img, meta = imageset[obj_index], metaset[obj_index]
    results = TSClassifier.predict({'image_input': img, 'meta_input': meta})

    print('Prediction: %s:%f, %s:%f, %s:%f\n' % (
    list(label_dict.keys())[0], results[0][0], list(label_dict.keys())[1], results[0][1], list(label_dict.keys())[2],
    results[0][2]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NEEDLE_training')
    parser.add_argument("-i", help="iteration with Makefile (Ignore)")
    args = vars(parser.parse_args())

    build_and_train_models(config.BAND, config.IMAGE_PATH, config.HOST_PATH, config.MAG_PATH, config.OUTPUT_PATH,
                           config.LABEL_PATH, config.QUALITY_MODEL_PATH, config.NO_DIFF, config.ONLY_COMPLETE, config.ADD_HOST, config.NEURONS,
                           config.RES_CNN_GROUP, config.BATCH_SIZE, config.EPOCH, config.LEARNING_RATE,
                           model_name='seed_' + str(config.SEED) + config.MODEL_NAME + args['i'])


