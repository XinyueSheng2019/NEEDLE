import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models

from keras.initializers import glorot_uniform

tf.random.set_seed(1)


class ResNetBlock(layers.Layer):
    def __init__(self, ks, filters, stage, s=1, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)

        self.s = s

        conv_name_base = 'res' + str(stage) + '_branch'
        bn_name_base = 'bn' + str(stage) + '_branch'

        F1, F2, F3 = filters


        self.Conv2D_1 = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))
        self.Batch_1 = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')
        self.Activation_1 = layers.Activation('relu')

        self.Conv2D_2 = layers.Conv2D(filters=F2, kernel_size=(ks, ks), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))
        self.Batch_2 = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')
        self.Activation_2 = layers.Activation('relu')

        self.Conv2D_3 = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))
        self.Batch_3 = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')

        self.shortcut_Conv2D = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))
        self.shortcut_Batch = layers.BatchNormalization(axis=3, name=bn_name_base + '1')

        self.Addcut = layers.Add()
        self.Activation_3 = layers.Activation('relu')


    def call(self, inputs):

        X_shortcut = inputs
        X = self.Conv2D_1(inputs)
        X = self.Batch_1(X)
        X = self.Activation_1(X)

        X = self.Conv2D_2(X)
        X = self.Batch_2(X)
        X = self.Activation_2(X)

        X = self.Conv2D_3(X)
        X = self.Batch_3(X)

        if self.s > 1 or inputs.shape[-1] != X.shape[-1]:
            X_shortcut = self.shortcut_Conv2D(X_shortcut)
            X_shortcut = self.shortcut_Batch(X_shortcut)

        X = self.Addcut([X, X_shortcut])

        return self.Activation_3(X)




class DataAugmentation(layers.Layer):
    def __init__(self, resize = 60, filp = "horizontal_and_vertical", rotation = 1, **kwargs):
        super().__init__()
        self.Resizing = layers.Resizing(resize, resize)
        self.RandomFlip = layers.RandomFlip(filp)
        self.RandomRotation = layers.RandomRotation([-1*rotation, 1*rotation], fill_mode = 'nearest')
    
    def call(self, inputs):
        X = self.Resizing(inputs)
        X = self.RandomFlip(X)
        X = self.RandomRotation(X)
        return X


class AttentionLayer(layers.Layer):
    pass





# data_augmentation = keras.Sequential([
#         layers.RandomFlip("horizontal"),
#         layers.RandomRotation([-1, 1], fill_mode = 'nearest'),
#         # layers.RandomBrightness(0.1)
#     ]
# )

# def identity_block(X, f, filters, stage, block):
   
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#     F1, F2, F3 = filters

#     X_shortcut = X
   
#     X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
#     X = Activation('relu')(X)

#     X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
#     X = Activation('relu')(X)

#     X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

#     X = Add()([X, X_shortcut])# SKIP Connection
#     X = Activation('relu')(X)

#     return X


# def convolutional_block(X, f, filters, stage, block, s=2):
   
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'

#     F1, F2, F3 = filters

#     X_shortcut = X

#     X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
#     X = Activation('relu')(X)

#     X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
#     X = Activation('relu')(X)

#     X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

#     X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
#     X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

#     X = Add()([X, X_shortcut])
#     X = Activation('relu')(X)

#     return X


# def short_cut(_input, residual):
#     """Adds a shortcut between input and residual block and merges them with "sum"
#     """
#     # Expand channels of shortcut to match residual.
#     # Stride appropriately to match residual (width, height)
#     # Should be int if network architecture is correctly configured.
#     ROW_AXIS = 1
#     COL_AXIS = 2
#     CHANNEL_AXIS = 3
#     input_shape = K.int_shape(_input)
#     residual_shape = K.int_shape(residual)
    
#     stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
#     stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))

#     equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]
    
#     shortcut = _input
#     if stride_width > 1 or stride_height > 1 or not equal_channels:
#         shortcut = layers.Conv2D(filters=residual_shape[CHANNEL_AXIS],
#                           kernel_size=(1, 1),
#                           strides=(stride_width, stride_height),
#                           padding="same")(_input)
    
#     print(_input.shape, shortcut.shape, residual.shape)
    
# #     kernel_initializer="he_normal",
# #                           kernel_regularizer=l2(0.0001)
        
#     return layers.Add()([shortcut, residual])