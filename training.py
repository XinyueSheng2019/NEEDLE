'''
https://www.tensorflow.org/guide/keras/custom_layers_and_models

'''

# set the seed
import tensorflow as tf
from numpy.random import seed
import config

tf.random.set_seed(config.SEED)
tf.keras.utils.set_random_seed(config.SEED)
# import random
# random.seed(config.SEED)
import os
# os.environ['PYTHONHASHSEED']=str(config.SEED)

from tensorflow.keras import backend as K


from tensorflow.keras.callbacks import EarlyStopping
import numpy as np 
import matplotlib.pyplot as plt
import csv
import json
from datetime import datetime
from transient_model import TransientClassifier, LossHistory

def train_with_kfold(train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels, label_dict, neurons= [128,128,128], res_cnn_group = None, batch_size = 32, epoch = 100, learning_rate = 0.00035, k_fold = True, n_splits = 5, model_name = None):
    unique, counts = np.unique(test_labels, return_counts=True)
    print('test: ', dict(zip(unique, counts)))

    unique, counts = np.unique(train_labels, return_counts=True)
    print('train: ', dict(zip(unique, counts)))


    current_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    
    class_weight = {}
    for i in np.arange(len(set(train_labels.flatten()))):
        class_weight[i] = train_labels.shape[0]/len(np.where(train_labels.flatten()==i)[0])
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = learning_rate,   #0.00035
                decay_steps=90,
                decay_rate=0.95,
                staircase=True)
    # lr_schedule = learning_rate

    earlystop = EarlyStopping(monitor = 'val_loss', patience = 3)
    split_list = StratifiedKFold(train_labels, n_splits = n_splits, shuffle = True)

    kfold_history = []
    idx = 1
    for train_idx, valid_idx in split_list:
        history = LossHistory()
        # build a classifier instance
        TCModel = TransientClassifier(label_dict, N_image = 60, dimension = train_imageset.shape[-1], neurons = neurons, Resnet_op = False)
        TCModel.build(input_shape = {'image_input':(None, train_imageset.shape[1], train_imageset.shape[2], train_imageset.shape[3]), 'meta_input': (None, train_metaset.shape[-1])})
        TCModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule), loss=tf.keras.losses.SparseCategoricalCrossentropy())
        TCModel.fit({'image_input': train_imageset[train_idx], 'meta_input': train_metaset[train_idx]}, train_labels[train_idx], shuffle = True, 
                    epochs=epoch, batch_size = batch_size, callbacks=[earlystop,history], class_weight = class_weight, 
                    validation_data = ({'image_input': train_imageset[valid_idx], 'meta_input': train_metaset[valid_idx]}, train_labels[valid_idx]))

        print('k-fold cross-validation, round ', idx, ':')
        TCModel.evaluate({'image_input': test_imageset, 'meta_input': test_metaset}, test_labels)
        kfold_history.append(history)
    
        # save the model and plots
        model_path = 'models_k_fold_' + current_time
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        TCModel.save(model_path + '/model_' + str(idx), save_format='tf')

        TCModel.plot_CM(test_imageset, test_metaset, test_labels, save_path = model_path)
        idx += 1
    
    print(kfold_history)
    plot_acc(kfold_history, model_path+'/val_acc.png')
    plot_loss(kfold_history,  model_path+'/val_loss.png')


def train(train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels, label_dict, neurons= [128,128,128], res_cnn_group = None, batch_size = 32, epoch = 100, learning_rate = 0.00035, model_name = None):

    unique, counts = np.unique(test_labels, return_counts=True)
    print('test: ', dict(zip(unique, counts)))

    unique, counts = np.unique(train_labels, return_counts=True)
    print('train: ', dict(zip(unique, counts)))


    current_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    
    class_weight = {}
    for i in np.arange(len(set(train_labels.flatten()))):
        class_weight[i] = train_labels.shape[0]/len(np.where(train_labels.flatten()==i)[0])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = learning_rate,   #0.00035
                decay_steps=100,
                decay_rate=0.95,
                staircase=True)
    # lr_schedule = learning_rate

    earlystop = EarlyStopping(monitor = 'val_loss', patience = 8)

    history = LossHistory()
    if res_cnn_group == None:
        TCModel = TransientClassifier(label_dict, N_image = 60, dimension = train_imageset.shape[-1], neurons = neurons )
    else:
        TCModel = TransientClassifier(label_dict, N_image = 60, dimension = train_imageset.shape[-1], neurons = neurons, res_cnn_group = res_cnn_group, Resnet_op = True)
    
    TCModel.build(input_shape = {'image_input':(None, train_imageset.shape[1], train_imageset.shape[2], train_imageset.shape[-1]), 'meta_input': (None, train_metaset.shape[-1])})
    TCModel.summary()
    
    
    # print initial paramters
    # print(TCModel.get_layer('conv_1').get_weights()[0])


    TCModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule), loss=tf.keras.losses.SparseCategoricalCrossentropy())
    TCModel.fit({'image_input': train_imageset, 'meta_input': train_metaset}, train_labels, shuffle = True, 
                    epochs=epoch, batch_size = batch_size, callbacks=[earlystop,history], class_weight = class_weight, validation_data = ({'image_input': test_imageset, 'meta_input': test_metaset}, test_labels)
                    )  
    TCModel.evaluate({'image_input': test_imageset, 'meta_input': test_metaset}, test_labels)
    if model_name is None:
        model_path = 'models/models_nor_' + current_time
    else:
        model_path = model_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    TCModel.save(model_path, save_format='tf')
    cm = TCModel.plot_CM(test_imageset, test_metaset, test_labels, save_path = model_path)
    cm_csv = model_path + '/results_cm.csv'
    with open(cm_csv, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(cm)
    f.close()

    # print(history)
    with open(model_path + '/loss_record.txt', 'w') as f:
        f.write(str(history.epoch_loss)+'\n'+str(history.epoch_val_loss)+'\n')
    f.close()


def StratifiedKFold(y, n_splits = 5, shuffle = True):
    '''
    Given the hybrid inputs, we have to custimise this algorithm.
    cross-validation:
    1. Take all of your labeled data, and divide it in K batches
    2. Train your model on K-1 batches
    3. Validate on the last, remaining batch
    4. Do this for all permutations

    return:
    a list of n_splits lists of index. seperated by n_splits - 1 and 1.
    '''

    n_samples = y.shape[-1]
    shuffle_idx = np.arange(n_samples)
    if shuffle == True:
        np.random.shuffle(shuffle_idx)
    shuffle_idx = list(shuffle_idx)
        
    split_sample_num = int(n_samples/n_splits)
    split_list = []
    
    for i in np.arange(n_splits-1):
        validate_sample = np.array(shuffle_idx[split_sample_num*i:split_sample_num*(i+1)])
        train_sample = np.array(shuffle_idx[:split_sample_num*i] + shuffle_idx[split_sample_num*(i+1):])

        split_list.append([train_sample, validate_sample])
    validate_sample = np.array(shuffle_idx[split_sample_num*(n_splits-1):])
    train_sample = np.array(shuffle_idx[:split_sample_num*(n_splits-1)])
    split_list.append([train_sample, validate_sample])

    return split_list


def plot_loss(khistory, save_path):
    plt.figure(figsize=(10, 6))
    k = 1
    for h in khistory:
        plt.plot(np.arange(len(h.epoch_val_loss)), h.epoch_val_loss, label = 'k' + str(k))
        k += 1
    plt.xlabel('epoch', fontsize = 16)
    plt.ylabel('val_loss', fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 14)

    plt.savefig(save_path)

def plot_acc(khistory, save_path):
    plt.figure(figsize=(10, 6))
    k = 1
    for h in khistory:
        plt.plot(np.arange(len(h.epoch_val_accuracy)), h.epoch_val_accuracy, label = 'k' + str(k))
        k += 1
    plt.xlabel('epoch', fontsize = 16)
    plt.ylabel('val_accuracy', fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 14)
    plt.savefig(save_path)



################### BUILD A MODEL #####################

# if __name__ == '__main__':


#     label_path = 'binary_label_dict.json'
#     label_dict = open(label_path,'r')
#     label_dict = json.loads(label_dict.read())

    # hash_path = 'hash_table.json'
    # hash_table = open(hash_path, 'r')
    # hash_table = json.loads(hash_table.read())

    # filepath = 'data/r_peak_set_check_bogus_20220118.hdf5'
    # model_name = 'c3_0118_r_without_bogus_TDE_SN'

    
    # train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels = preprocessing(filepath, label_dict, hash_table)
   

    # train(train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels, label_dict["label"], batch_size = 32, epoch = 300, learning_rate = 8e-5, k_fold = False, n_splits = 3, model_name = model_name)


