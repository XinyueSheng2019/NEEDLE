'''
This file is to classify SN, SLSN-I, TDE only based on their host photometry and offset.
'''
# set the seed
import tensorflow as tf
from numpy.random import seed
import json
import config
seed(config.SEED)
tf.random.set_seed(config.SEED)
tf.keras.utils.set_random_seed(config.SEED)

import os, json
from build_dataset import single_band_peak_db
from preprocessing import preprocessing, cut_preprocessing, single_transient_preprocessing,open_with_h5py

import os
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping


SEED = 456
BAND = 'r'
IMAGE_PATH = '../../../data/image_sets_v3'
HOST_PATH = '../../../data/host_info_r5'
MAG_PATH = '../../../data/mag_sets_v4'
OUTPUT_PATH = '../model_with_data/' + BAND + '_band/only_host_nor_20231128/'
LABEL_PATH = '../model_labels/label_dict_equal_test.json'
QUALITY_MODEL_PATH = '../../bogus_classifier/models/bogus_model_without_zscale'
NO_DIFF = True
ONLY_COMPLETE = True
NEURONS = [[128,3],[128,3]]
RES_CNN_GROUP = None
BATCH_SIZE = 128
EPOCH = 300
LEARNING_RATE = 3e-5



def plot_CM(self, test_meta, test_labels, save_path, label_dict):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    predictions = self.predict(test_meta, batch_size= 1)
    # self.evaluate()
    y_pred = np.argmax(predictions, axis=-1)
    y_true = test_labels.flatten()
    cm = confusion_matrix(y_true, y_pred)

    p_cm = []
    for i in cm:
        p_cm.append(np.round(i/np.sum(i),3))
    # cm = p_cm

    ## Get Class Labels
    labels = label_dict.keys()
    class_names = labels

    # Plot confusion matrix in a beautiful manner
    fig = plt.figure(figsize=(16, 14))
    ax= plt.subplot()
    sns.heatmap(p_cm, annot=True, ax = ax, fmt = 'g', annot_kws={"size": 20}); #annot=True to annotate cells
        
    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=30)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(class_names, fontsize = 20)
    ax.xaxis.tick_bottom()
    ax.set_ylabel('True', fontsize=30)
    ax.yaxis.set_ticklabels(class_names, fontsize = 20)
    ax.tick_params(labelsize=20)
    plt.yticks(rotation=0)

    current_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    # plt.title('Confusion Matrix ', fontsize=20)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(save_path+'/cm_'+ current_time +'.png')

    return cm


for i in np.arange(10):
    MODEL_NAME = 'host_model_' + str(i)

    label_dict = open(LABEL_PATH,'r')
    label_dict = json.loads(label_dict.read())
    # print(label_dict)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        BClassifier = models.load_model(QUALITY_MODEL_PATH) 
        single_band_peak_db(IMAGE_PATH, HOST_PATH, MAG_PATH, OUTPUT_PATH, label_dict["classify"], band = BAND, no_diff= NO_DIFF, only_complete = ONLY_COMPLETE, add_host=True, BClassifier = BClassifier)
    else:
        print('Data already exist! Start training.\n')

    filepath = OUTPUT_PATH + 'data.hdf5'
    hash_path = OUTPUT_PATH + 'hash_table.json'
    model_path = OUTPUT_PATH + MODEL_NAME

    _, train_metaset, train_labels, _, test_metaset, test_labels = preprocessing(filepath, label_dict, hash_path, model_path)

    # train_metaset, test_metaset = train_metaset[:,7:], test_metaset[:,7:]
    print(train_metaset.shape)
    

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape = (None, 8)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    earlystop = EarlyStopping(monitor = 'val_loss', patience = 8)
    class_weight = {}
    for i in np.arange(len(set(train_labels.flatten()))):
        class_weight[i] = train_labels.shape[0]/len(np.where(train_labels.flatten()==i)[0])

    model.build()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 2e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy())
    model.fit(train_metaset, train_labels, batch_size = 128, callbacks=[earlystop], class_weight = class_weight, epochs = 1000, validation_data = (test_metaset, test_labels))
    plot_CM(model, test_metaset, test_labels, save_path = model_path, label_dict=label_dict["label"])
    model.save(model_path, save_format='tf')






