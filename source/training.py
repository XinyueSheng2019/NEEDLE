import os
from datetime import datetime
import csv
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from transient_model import TransientClassifier, LossHistory

def train_with_kfold(train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels, label_dict, neurons=[128, 128, 128], batch_size=32, epoch=100, learning_rate=0.00035, n_splits=5, model_name=None):
    unique, counts = np.unique(test_labels, return_counts=True)
    print('Test labels distribution:', dict(zip(unique, counts)))

    unique, counts = np.unique(train_labels, return_counts=True)
    print('Train labels distribution:', dict(zip(unique, counts)))

    current_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    class_weight = {}
    for i in np.unique(train_labels):
        class_weight[i] = train_labels.shape[0] / len(np.where(train_labels == i)[0])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=90,
        decay_rate=0.95,
        staircase=True)

    early_stop = EarlyStopping(monitor='val_loss', patience=3)

    split_list = StratifiedKFold(train_labels, n_splits=n_splits, shuffle=True)

    kfold_history = []
    idx = 1
    for train_idx, valid_idx in split_list:
        history = LossHistory()
        model = TransientClassifier(label_dict, N_image=60, dimension=train_imageset.shape[-1], neurons=neurons, Resnet_op=False)
        model.build(input_shape={'image_input': (None, train_imageset.shape[1], train_imageset.shape[2], train_imageset.shape[3]), 'meta_input': (None, train_metaset.shape[-1])})
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=tf.keras.losses.SparseCategoricalCrossentropy())
        model.fit({'image_input': train_imageset[train_idx], 'meta_input': train_metaset[train_idx]}, train_labels[train_idx], shuffle=True, epochs=epoch, batch_size=batch_size, callbacks=[early_stop, history], class_weight=class_weight, validation_data=({'image_input': train_imageset[valid_idx], 'meta_input': train_metaset[valid_idx]}, train_labels[valid_idx]))

        print('K-Fold cross-validation, round {}:'.format(idx))
        model.evaluate({'image_input': test_imageset, 'meta_input': test_metaset}, test_labels)
        kfold_history.append(history)

        model_path = 'models_k_fold_' + current_time
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save(model_path + '/model_' + str(idx), save_format='tf')
        model.plot_CM(test_imageset, test_metaset, test_labels, save_path=model_path)
        idx += 1

    plot_acc(kfold_history, model_path + '/val_acc.png')
    plot_loss(kfold_history, model_path + '/val_loss.png')

def train(train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels, label_dict, neurons=[128, 128, 128], batch_size=32, epoch=100, learning_rate=0.00035, model_name=None):
    unique, counts = np.unique(test_labels, return_counts=True)
    print('Test labels distribution:', dict(zip(unique, counts)))

    unique, counts = np.unique(train_labels, return_counts=True)
    print('Train labels distribution:', dict(zip(unique, counts)))

    current_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    class_weight = {}
    for i in np.unique(train_labels):
        class_weight[i] = train_labels.shape[0] / len(np.where(train_labels == i)[0])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=100,
        decay_rate=0.95,
        staircase=True)

    early_stop = EarlyStopping(monitor='val_loss', patience=8)

    history = LossHistory()
    model = TransientClassifier(label_dict, N_image=60, dimension=train_imageset.shape[-1], neurons=neurons)
    model.build(input_shape={'image_input': (None, train_imageset.shape[1], train_imageset.shape[2], train_imageset.shape[-1]), 'meta_input': (None, train_metaset.shape[-1])})
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=tf.keras.losses.SparseCategoricalCrossentropy())
    model.fit({'image_input': train_imageset, 'meta_input': train_metaset}, train_labels, shuffle=True, epochs=epoch, batch_size=batch_size, callbacks=[early_stop, history], class_weight=class_weight, validation_data=({'image_input': test_imageset, 'meta_input': test_metaset}, test_labels))

    model.evaluate({'image_input': test_imageset, 'meta_input': test_metaset}, test_labels)

    if model_name is None:
        model_path = 'models/models_nor_' + current_time
    else:
        model_path = model_name

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model.save(model_path, save_format='tf')

    cm = model.plot_CM(test_imageset, test_metaset, test_labels, save_path=model_path)
    cm_csv = model_path + '/results_cm.csv'
    with open(cm_csv, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(cm)
    f.close()

    with open(model_path + '/loss_record.txt', 'w') as f:
        f.write(str(history.epoch_loss) + '\n' + str(history.epoch_val_loss) + '\n')
    f.close()

def plot_loss(khistory, save_path):
    plt.figure(figsize=(10, 6))
    for k, h in enumerate(khistory, 1):
        plt.plot(np.arange(len(h.epoch_val_loss)), h.epoch_val_loss, label='k{}'.format(k))
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Validation Loss', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(save_path)

def plot_acc(khistory, save_path):
    plt.figure(figsize=(10, 6))
    for k, h in enumerate(khistory, 1):
        plt.plot(np.arange(len(h.epoch_val_accuracy)), h.epoch_val_accuracy, label='k{}'.format(k))
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Validation Accuracy', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(save_path)
