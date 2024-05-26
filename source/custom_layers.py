import tensorflow as tf
from tensorflow.keras import layers

tf.random.set_seed(1)


class ResNetBlock(layers.Layer):
    def __init__(self, ks, filters, stage, s=1, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.s = s
        conv_name_base = 'res' + str(stage) + '_branch'
        bn_name_base = 'bn' + str(stage) + '_branch'
        F1, F2, F3 = filters
        
        # First convolution block
        self.Conv2D_1 = layers.Conv2D(F1, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a')
        self.Batch_1 = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')
        self.Activation_1 = layers.Activation('relu')
        
        # Second convolution block
        self.Conv2D_2 = layers.Conv2D(F2, (ks, ks), padding='same', name=conv_name_base + '2b')
        self.Batch_2 = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')
        self.Activation_2 = layers.Activation('relu')
        
        # Third convolution block
        self.Conv2D_3 = layers.Conv2D(F3, (1, 1), padding='valid', name=conv_name_base + '2c')
        self.Batch_3 = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')
        
        # Shortcut connection
        self.shortcut_Conv2D = layers.Conv2D(F3, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1')
        self.shortcut_Batch = layers.BatchNormalization(axis=3, name=bn_name_base + '1')
        
        self.Add = layers.Add()
        self.Activation_3 = layers.Activation('relu')

    def call(self, inputs):
        X_shortcut = inputs
        
        # Main path
        X = self.Conv2D_1(inputs)
        X = self.Batch_1(X)
        X = self.Activation_1(X)

        X = self.Conv2D_2(X)
        X = self.Batch_2(X)
        X = self.Activation_2(X)

        X = self.Conv2D_3(X)
        X = self.Batch_3(X)
        
        # Shortcut path
        if self.s > 1 or inputs.shape[-1] != X.shape[-1]:
            X_shortcut = self.shortcut_Conv2D(X_shortcut)
            X_shortcut = self.shortcut_Batch(X_shortcut)
        
        # Add shortcut and main path
        X = self.Add([X, X_shortcut])
        return self.Activation_3(X)


class DataAugmentation(layers.Layer):
    def __init__(self, resize=60, flip="horizontal_and_vertical", rotation=1, **kwargs):
        super().__init__(**kwargs)
        self.Resizing = layers.Resizing(resize, resize)
        self.RandomFlip = layers.RandomFlip(flip)
        self.RandomRotation = layers.RandomRotation([-1 * rotation, 1 * rotation], fill_mode='nearest')

    def call(self, inputs):
        X = self.Resizing(inputs)
        X = self.RandomFlip(X)
        X = self.RandomRotation(X)
        return X

