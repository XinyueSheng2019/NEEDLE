import tensorflow as tf
from numpy.random import seed

# Set the random seeds for reproducibility
import sys
sys.path.append('../source')
import config
seed(config.SEED)
tf.random.set_seed(config.SEED)
tf.keras.utils.set_random_seed(config.SEED)

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras import layers, backend as K
from custom_layers import ResNetBlock, DataAugmentation
from sklearn.metrics import confusion_matrix
import seaborn as sns

class TransientClassifier(tf.keras.Model):
    """
    A custom keras model for building a transient CNN-based classifier. 
    ResNet block or plain CNN are both available.
    Metadata are also added.
    """

    def __init__(self, label_dict, N_image, dimension, meta_dimension=11, ks=16, pooling_size=2, neurons=None, 
                 res_cnn_group=None, Resnet_op=False, meta_only=False, **kwargs):
        super(TransientClassifier, self).__init__(**kwargs)
        
        if neurons is None:
            neurons = [[128, 2], [128, 2], [128, 2]]
        
        self.N_image = N_image
        self.dimension = dimension
        self.meta_dimension = meta_dimension
        self.ks = ks
        self.pooling_size = pooling_size
        self.neurons = neurons
        self.res_group = res_cnn_group 
        self.label_dict = label_dict
        self.Resnet_op = Resnet_op
        self.meta_only = meta_only

        # Data Augmentation Layer
        self.data_augmentation = DataAugmentation()

        # Image input and CNN layers
        self.image_input = layers.Input(shape=(N_image, N_image, dimension), name='image_input')
        self.conv2d_1 = layers.Conv2D(neurons[0][0], 5, activation='relu', name='conv_1')
        self.pooling_1 = layers.MaxPooling2D(pooling_size, pooling_size)
        self.conv2d_2 = layers.Conv2D(neurons[1][0], 3, activation='relu', name='conv_2')
        self.pooling_2 = layers.MaxPooling2D(pooling_size, pooling_size)

        if Resnet_op:
            self.res_block = ResNetBlock(ks=ks, filters=res_cnn_group, stage=1, s=1)
        else:
            self.cnn_layers = []
            for cy in neurons[1:]:
                conv2d = layers.Conv2D(cy[0], 3, activation='relu', name='conv_3')
                pooling = layers.MaxPooling2D(cy[1], cy[1])
                self.cnn_layers.append((conv2d, pooling))

        self.flatten = layers.Flatten()

        # Metadata input and dense layers
        self.meta_input = layers.Input(shape=(meta_dimension), name='meta_input')
        self.dense_m1 = layers.Dense(128, activation='relu', name='dense_me1')
        self.dense_m2 = layers.Dense(128, activation='relu', name='dense_me2')

        # Combined model dense layers
        self.concatenate = layers.Concatenate(axis=-1, name='concatenate')
        self.dense_c1 = layers.Dense(256, activation='relu', name='dense_c1')
        self.dense_c2 = layers.Dense(32, activation='relu', name='dense_c2')
        self.output_layer = layers.Dense(len(label_dict), activation='softmax', name='output')

    def call(self, inputs):
        if not self.meta_only:
            x = self.data_augmentation(inputs['image_input'])
            x = self.conv2d_1(x)
            x = self.pooling_1(x)
            x = self.conv2d_2(x)
            x = self.pooling_2(x)

            if self.Resnet_op:
                x = self.res_block(x)
            else:
                for conv2d, pooling in self.cnn_layers:
                    x = conv2d(x)
                    x = pooling(x)

            x = self.flatten(x)
            y = self.dense_m1(inputs['meta_input'])
            y = self.dense_m2(y)
            z = self.concatenate([x, y])
            z = self.dense_c1(z)
            z = self.dense_c2(z)
            return self.output_layer(z)
        else:
            y = self.dense_m1(inputs['meta_input'])
            y = self.dense_m2(y)
            z = self.dense_c1(y)
            z = self.dense_c2(z)
            return self.output_layer(z)

    def plot_CM(self, test_images, test_meta, test_labels, save_path, suffix=''):
        
        predictions = self.predict({'image_input': test_images, 'meta_input': test_meta}, batch_size=1)
        y_pred = np.argmax(predictions, axis=-1)
        y_true = test_labels.flatten()
        cm = confusion_matrix(y_true, y_pred)

        p_cm = np.round(cm / np.sum(cm, axis=1, keepdims=True), 3)

        labels = self.label_dict.keys()
        class_names = labels

        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(p_cm, annot=True, ax=ax, fmt='g', annot_kws={"size": 20})
          
        ax.set_xlabel('Predicted', fontsize=30)
        ax.xaxis.set_label_position('bottom')
        plt.xticks(rotation=90)
        ax.xaxis.set_ticklabels(class_names, fontsize=20)
        ax.xaxis.tick_bottom()
        ax.set_ylabel('True', fontsize=30)
        ax.yaxis.set_ticklabels(class_names, fontsize=20)
        ax.tick_params(labelsize=20)
        plt.yticks(rotation=0)

        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        plt.savefig(os.path.join(save_path, f'cm_{suffix}_{current_time}.png'))

        return cm


class LossHistory(tf.keras.callbacks.Callback):
    """
    This class is used for recording the loss, accuracy, AUC, f1_score value during training.
    """
    def on_train_begin(self, logs=None):
        self.epoch_loss = []
        self.epoch_accuracy = []
        self.epoch_val_loss = []
        self.epoch_val_accuracy = []
        self.batch_losses = []
        self.batch_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_loss.append(logs.get('loss'))
        self.epoch_accuracy.append(logs.get('accuracy'))
        self.epoch_val_loss.append(logs.get('val_loss'))
        self.epoch_val_accuracy.append(logs.get('val_accuracy'))

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))
        self.batch_accuracy.append(logs.get('accuracy'))

    def save_to_json(self, file_path):
        """
        Save the recorded loss and accuracy to a JSON file.
        """
        history_dict = {
            "epoch_loss": self.epoch_loss,
            "epoch_val_loss": self.epoch_val_loss,
            "batch_losses": self.batch_losses,
        }
        with open(file_path, 'w') as f:
            json.dump(history_dict, f, indent=4)