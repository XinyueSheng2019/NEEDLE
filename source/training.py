import os
import sys
sys.path.append('../')
import tensorflow as tf
import config
tf.random.set_seed(config.SEED)
tf.keras.utils.set_random_seed(config.SEED)

import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping
from transient_model import TransientClassifier, LossHistory



def stratified_kfold(y, n_splits=5, shuffle=True):
    """
    Perform stratified k-fold cross-validation split.

    Parameters:
        y (np.array): Labels of the dataset.
        n_splits (int): Number of splits.
        shuffle (bool): Whether to shuffle the dataset before splitting.

    Returns:
        list: List of tuples with train and validation indices.
    """
    n_samples = y.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    split_size = n_samples // n_splits
    splits = []

    for i in range(n_splits):
        val_idx = indices[i * split_size: (i + 1) * split_size]
        train_idx = np.concatenate([indices[:i * split_size], indices[(i + 1) * split_size:]])
        splits.append((train_idx, val_idx))
    
    return splits


def train_with_kfold(train_images, train_meta, train_labels, test_images, test_meta, test_labels, label_dict,
                     neurons=[128, 128, 128], batch_size=32, epochs=100, learning_rate=0.00035, n_splits=5, model_name=None):
    """
    Train the model using stratified k-fold cross-validation.

    Parameters:
        Various datasets and training parameters.
    """
    unique, counts = np.unique(test_labels, return_counts=True)
    print('Test label distribution:', dict(zip(unique, counts)))

    unique, counts = np.unique(train_labels, return_counts=True)
    print('Train label distribution:', dict(zip(unique, counts)))

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    class_weight = {i: train_labels.shape[0] / len(np.where(train_labels.flatten() == i)[0])
                    for i in range(len(set(train_labels.flatten())))}
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=90,
        decay_rate=0.95,
        staircase=True
    )

    earlystop = EarlyStopping(monitor='val_loss', patience=3)
    split_list = stratified_kfold(train_labels, n_splits=n_splits, shuffle=True)

    kfold_history = []
    for idx, (train_idx, val_idx) in enumerate(split_list, start=1):
        history = LossHistory()

        # Build and compile the model
        TCModel = TransientClassifier(label_dict, N_image=60, dimension=train_images.shape[-1], neurons=neurons)
        TCModel.build(input_shape={
            'image_input': (None, train_images.shape[1], train_images.shape[2], train_images.shape[3]),
            'meta_input': (None, train_meta.shape[-1])
        })
        TCModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy())
        
        TCModel.fit(
            {'image_input': train_images[train_idx], 'meta_input': train_meta[train_idx]},
            train_labels[train_idx],
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[earlystop, history],
            class_weight=class_weight,
            validation_data=(
                {'image_input': train_images[val_idx], 'meta_input': train_meta[val_idx]},
                train_labels[val_idx]
            )
        )

        print(f'K-fold cross-validation, round {idx}:')
        TCModel.evaluate({'image_input': test_images, 'meta_input': test_meta}, test_labels)
        kfold_history.append(history)
        
        # Save the model and plots
        model_path = f'models_k_fold_{current_time}'
        os.makedirs(model_path, exist_ok=True)
        TCModel.save(os.path.join(model_path, f'model_{idx}'), save_format='tf')
        TCModel.plot_CM(test_images, test_meta, test_labels, save_path=model_path)
    
    plot_acc(kfold_history, os.path.join(model_path, 'val_acc.png'))
    plot_loss(kfold_history, os.path.join(model_path, 'val_loss.png'))


def train(train_images, train_meta, train_labels, test_images, test_meta, test_labels, label_dict,
          neurons=[128, 128, 128], res_cnn_group=None, meta_only=False, batch_size=32, epochs=100,
          learning_rate=0.00035, model_name=None):
    """
    Train the model on the entire training set without cross-validation.

    Parameters:
        Various datasets and training parameters.
    """
    unique, counts = np.unique(test_labels, return_counts=True)
    print('Test label distribution:', dict(zip(unique, counts)))

    unique, counts = np.unique(train_labels, return_counts=True)
    print('Train label distribution:', dict(zip(unique, counts)))

    print(train_images.shape, train_meta.shape)

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    class_weight = {i: train_labels.shape[0] / len(np.where(train_labels.flatten() == i)[0])
                    for i in range(len(set(train_labels.flatten())))}

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=100,
        decay_rate=0.95,
        staircase=True
    )

    earlystop = EarlyStopping(monitor='val_loss', patience=8)
    history = LossHistory()

    TCModel = TransientClassifier(label_dict, N_image=60, dimension=train_images.shape[-1], neurons=neurons,
                                  meta_only=meta_only, res_cnn_group=res_cnn_group, Resnet_op=res_cnn_group is not None)

    TCModel.build(input_shape={
        'image_input': (None, train_images.shape[1], train_images.shape[2], train_images.shape[-1]),
        'meta_input': (None, train_meta.shape[-1])
    })
    TCModel.summary()

    TCModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy())

    TCModel.fit(
        {'image_input': train_images, 'meta_input': train_meta},
        train_labels,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[earlystop, history],
        class_weight=class_weight,
        validation_data=({'image_input': test_images, 'meta_input': test_meta}, test_labels),
        use_multiprocessing=True
    )
    
    TCModel.evaluate({'image_input': test_images, 'meta_input': test_meta}, test_labels)

    model_path = model_name if model_name else f'models/models_nor_{current_time}'
    os.makedirs(model_path, exist_ok=True)
    TCModel.save(model_path, save_format='tf')

    cm = TCModel.plot_CM(test_images, test_meta, test_labels, save_path=model_path)
    with open(os.path.join(model_path, 'results_cm.csv'), 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(cm)

    history.save_to_json(os.path.join(model_path, 'loss_records.json'))

def plot_loss(history_list, save_path):
    """
    Plot validation loss across k-folds.

    Parameters:
        history_list (list): List of LossHistory objects.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    for k, history in enumerate(history_list, start=1):
        plt.plot(np.arange(len(history.epoch_val_loss)), history.epoch_val_loss, label=f'k{k}')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Validation Loss', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(save_path)


def plot_acc(history_list, save_path):
    """
    Plot validation accuracy across k-folds.

    Parameters:
        history_list (list): List of LossHistory objects.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    for k, history in enumerate(history_list, start=1):
        plt.plot(np.arange(len(history.epoch_val_accuracy)), history.epoch_val_accuracy, label=f'k{k}')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Validation Accuracy', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(save_path)


################### MAIN EXECUTION #####################

# if __name__ == '__main__':
#     import json

#     label_path = 'binary_label_dict.json'
#     with open(label_path, 'r') as file:
#         label_dict = json.load(file)

#     hash_path = 'hash_table.json'
#     with open(hash_path, 'r') as file:
#         hash_table = json.load(file
