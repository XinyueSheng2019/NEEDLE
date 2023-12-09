# set the seed
import tensorflow as tf
from numpy.random import seed

import config
seed(config.SEED)
tf.random.set_seed(config.SEED)
tf.keras.utils.set_random_seed(config.SEED)
# import random
# random.seed(config.SEED)

import os
# os.environ['PYTHONHASHSEED']=str(config.SEED)

from tensorflow.keras import backend as K
# config_s = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
# tf.set_random_seed(sd)
# sess = tf.Session(graph=tf.get_default_graph(), config=config_s)
# K.set_session(sess)


from tensorflow.keras import layers

from custom_layers import ResNetBlock, DataAugmentation

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
from datetime import datetime



class TransientClassifier(tf.keras.Model):
    '''
    A custom keras model for building a transient CNN-based classifier. 
    ResNet block or plain CNN are both available.
    Metadata are also added.
    '''

    def __init__(self, label_dict, N_image, dimension, meta_dimension = 11, ks = 16, pooling_size = 2, neurons = [[128,2],[128,2],[128,2]], res_cnn_group = None, Resnet_op = False, **kwargs):
        super(TransientClassifier, self).__init__(**kwargs)
        self.N_image = N_image
        self.ks = ks
        self.pooling_size = pooling_size
        self.dimension = dimension
        self.meta_dimension = meta_dimension
        self.res_group = res_cnn_group 
        self.label_dict = label_dict
        self.Resnet_op = Resnet_op
        self.neurons = neurons
        self.cnn_layers = []

        self.DataAugmentation = DataAugmentation()
        self.input1 = layers.Input(shape=(self.N_image, self.N_image, self.dimension), name = 'image_input')
        self.Conv2D_1 = layers.Conv2D(self.neurons[0][0], 3, activation='relu', name = 'conv_1')
        self.pooling_1 = layers.MaxPooling2D((self.neurons[0][1],self.neurons[0][1]))
        self.Conv2D_2 = layers.Conv2D(self.neurons[1][0], 3, activation='relu', name = 'conv_2')
        self.pooling_2 = layers.MaxPooling2D((self.neurons[1][1],self.neurons[1][1]))


        if self.Resnet_op == True:
            self.block_1 = ResNetBlock(ks = self.ks, filters = self.res_group, stage = 1, s=1)
        else:
            for cy in self.neurons[1:]:
                Conv2D = layers.Conv2D(cy[0], 3, activation='relu', name = 'conv_3')
                pooling = layers.MaxPooling2D((cy[1],cy[1]))
                self.cnn_layers.append([Conv2D, pooling])
    
        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(64, activation='relu', name = 'dense_im1')

        #### metadata dense layer
        # self.input2 = layers.Input(shape=(self.meta_dimension), name = 'meta_input')
        self.normalization = layers.LayerNormalization(axis=1)
        self.dense_m1 = layers.Dense(units = 128, activation = 'relu', name = 'dense_me1', input_dim = self.meta_dimension)
        self.dense_m2 = layers.Dense(units = 128, activation = 'relu', name = 'dense_me2')

        ### combine two sub-models
        self.concatenate = layers.Concatenate(axis = -1, name = 'concatenate')
        self.dense_c1 = layers.Dense(192,  activation = 'relu', name = 'dense_c1') #kernel_initializer = 'uniform',
        self.dense_c2 = layers.Dense(32,  activation = 'relu', name = 'dense_c2')
        self.dense_2 = layers.Dense(len(self.label_dict), activation='softmax' , name = 'output')

    def call(self, inputs):
 
        X = self.DataAugmentation(inputs['image_input'])
        X = self.Conv2D_1(X)
        X = self.pooling_1(X)
        X = self.Conv2D_2(X)
        X = self.pooling_2(X)
        if self.Resnet_op == True:
            X = self.block_1(X)
        else:
            for cy in self.cnn_layers:
                X = cy[0](X)
                X = cy[1](X)

        X = self.flatten(X)
        # Y = self.normalization(inputs['meta_input'])
        Y = self.dense_m1(inputs['meta_input'])
        Y = self.dense_m2(Y)
        Z = self.concatenate([X, Y])
    
        Z = self.dense_c1(Z)
        Z = self.dense_c2(Z)

        return self.dense_2(Z)

    def plot_CM(self, test_images, test_meta, test_labels, save_path):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        predictions = self.predict({'image_input': test_images, 'meta_input': test_meta}, batch_size= 1)
        # self.evaluate()
        y_pred = np.argmax(predictions, axis=-1)
        y_true = test_labels.flatten()
        cm = confusion_matrix(y_true, y_pred)

        p_cm = []
        for i in cm:
            p_cm.append(np.round(i/np.sum(i),3))
        # cm = p_cm

        ## Get Class Labels
        labels = self.label_dict.keys()
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



class LossHistory(tf.keras.callbacks.Callback):
    '''
    This class is used for recording the loss, accuracy, AUC, f1_score value during training.
    '''
    def on_train_begin(self, logs={}):
        self.epoch_loss = []
        self.epoch_accuracy = []

        self.epoch_val_loss = []
        self.epoch_val_accuracy = []
 
        self.batch_losses = []
        self.batch_accuracy = []

      

    def on_epoch_end(self, batch, logs={}):
        self.epoch_loss.append(logs.get('loss'))
        self.epoch_accuracy.append(logs.get('accuracy'))
       
        self.epoch_val_loss.append(logs.get('val_loss'))
        self.epoch_val_accuracy.append(logs.get('val_accuracy'))
    

    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))
        self.batch_accuracy.append(logs.get('accuracy'))
   

