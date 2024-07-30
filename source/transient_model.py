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
from tensorflow.keras.layers import Layer
from custom_layers import ResNetBlock, DataAugmentation
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras import models
import wandb


class EM_QualityClassifier():
    def __init__(self, model_path, iter = 5):
        self.models = []
        for i in np.arange(iter):
            bcmodel = models.load_model(os.path.join(model_path, f'quality_model_20140524_{i}'))
            self.models.append(bcmodel)
    def predict(self, img, threshold = 0.75):
        img = img.reshape(1, 60, 60, 1)
        results = []
        for m in self.models:
            results.append(m.predict(img)[0][1])
        result = np.mean(results, axis = 0)
        return result >= threshold
    
class FeatureWeightedLayer(Layer):
    def __init__(self, feature_weights, **kwargs):
        super(FeatureWeightedLayer, self).__init__(**kwargs)
        self.feature_weights = np.array(feature_weights)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1],),
                                      initializer='uniform',
                                      trainable=True)

    def call(self, inputs):
        return inputs * self.feature_weights


class TransientClassifier(tf.keras.Model):
    """
    A custom keras model for building a transient CNN-based classifier. 
    ResNet block or plain CNN are both available.
    Metadata are also added.
    """

    def __init__(self, label_dict, N_image, image_dimension, meta_dimension, ks=16, neurons=None, 
                 res_cnn_group=None, Resnet_op=False, meta_only=False, feature_importance = np.array,  **kwargs):
        super(TransientClassifier, self).__init__(**kwargs)
        
        if neurons is None:
            neurons = [[64, 5], [128, 3]]
        
        self.N_image = N_image
        self.image_dimension = image_dimension
        self.meta_dimension = meta_dimension
        self.ks = ks
        self.neurons = neurons
        self.res_group = res_cnn_group 
        self.label_dict = label_dict
        self.Resnet_op = Resnet_op
        self.meta_only = meta_only
        self.cnn_layers = []
        self.feature_importance = feature_importance

        # Data Augmentation Layer
        self.data_augmentation = DataAugmentation()

        # Image input and CNN layers
        self.image_input = layers.Input(shape=(N_image, N_image, self.image_dimension), name='image_input')
        for i in np.arange(len(neurons)):
            self.cnn_layers.append([layers.Conv2D(neurons[i][0], 3, activation='relu', name=f'conv_{i}'), layers.MaxPooling2D((self.neurons[i][1],self.neurons[i][1]), name = f'pool_{i}')])

        if Resnet_op:
            self.res_block = ResNetBlock(ks=ks, filters=res_cnn_group, stage=1, s=1)

        self.flatten = layers.Flatten()

        # Metadata input and dense layers
        self.meta_input = layers.Input(shape=(meta_dimension), name='meta_input')
        if self.feature_importance is not None:
            self.meta_weighted = FeatureWeightedLayer(self.feature_importance, name = 'feature_ranking')
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

            if self.Resnet_op:
                x = self.res_block(x)
            else:
                for conv2d, pooling in self.cnn_layers:
                    x = conv2d(x)
                    x = pooling(x)

            x = self.flatten(x)
            if self.feature_importance is not None:
                y = self.meta_weighted(inputs['meta_input'])
                y = self.dense_m1(y)
            else:
                y = self.dense_m1(inputs['meta_input'])
            y = self.dense_m2(y)
            z = self.concatenate([x, y])
            z = self.dense_c1(z)
            z = self.dense_c2(z)
            return self.output_layer(z)
        else:
            y = self.meta_weighted(inputs['meta_input'])
            y = self.dense_m1(y)
            y = self.dense_m2(y)
            z = self.dense_c1(y)
            z = self.dense_c2(z)
            return self.output_layer(z)

    def get_config(self):
            config = super(TransientClassifier, self).get_config()
            return config
    def plot_CM(self, test_images, test_meta, test_labels, save_path, suffix=''):
        
        predictions = self.predict({'image_input': test_images, 'meta_input': test_meta}, batch_size=1)
        y_pred = np.argmax(predictions, axis=-1)
        y_true = test_labels.flatten()

        labels = self.label_dict.keys()
        class_names = list(labels)

        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=y_true, preds=y_pred,
                        class_names=class_names)})


        cm = confusion_matrix(y_true, y_pred)

        p_cm = np.round(cm / np.sum(cm, axis=1, keepdims=True), 3)

        

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
