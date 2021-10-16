import numpy as np 
import tensorflow as tf 
from tensorflow.keras import models, layers 

class EncodeLayer(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1): 
        super(EncodeLayer, self).__init__()

        self.conv1 = layers.Conv2D(filters, kernel_size, strides, padding)
        self.conv2 = layers.Conv2D(filters, kernel_size, strides, padding)
        self.relu1 = layers.LeakyReLU(alpha=0.2)
        self.relu2 = layers.LeakyReLU(alpha=0.2)
        self.bnorm1 = layers.BatchNormalization() 
        self.bnorm2 = layers.BatchNormalization() 

        self.pooling = layers.MaxPool2D(2, 2)

    def call(self, input_tensor):
        x1 = self.conv1(input_tensor) 
        x2 = self.bnorm1(x1)
        x2 = self.relu1(x2) 
        
        x2 = self.conv2(x2)
        x2 = self.bnorm2(x2)
        x2 = self.relu2(x2) 
        
        x2 += x1 
        x_out = self.pooling(x2) 
        return x2, x_out 


class DecodeLayer(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1):
        super(DecodeLayer, self).__init__() 

        self.upsample = layers.UpSampling2D((2, 2)) 
        self.conv1 = layers.Conv2D(filters, kernel_size, strides, padding)
        self.conv2 = layers.Conv2D(filters, kernel_size, strides, padding)
        self.relu1 = layers.LeakyReLU(alpha=0.2)
        self.relu2 = layers.LeakyReLU(alpha=0.2)
        self.bnorm1 = layers.BatchNormalization() 
        self.bnorm2 = layers.BatchNormalization()

        self.concat = layers.Concatenate() 

    def call(self, input_tensor, skip):
        x = self.upsample(input_tensor) 
        concat = self.concat([x, skip])
        x = self.conv1(concat)
        x = self.bnorm1(x)
        x = self.relu1(x) 

        x = self.conv2(x)
        x = self.bnorm(x)
        x = self.relu2(x)

        return x 


class BottleNeck(layers.Layer): 
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1): 
        super(BottleNeck, self).__init__() 
        self.conv1 = layers.Conv2D(filters, kernel_size, strides, padding)
        self.conv2 = layers.Conv2D(filters, kernel_size, strides, padding)
        self.relu1 = layers.LeakyReLU(alpha=0.2)
        self.relu2 = layers.LeakyReLU(alpha=0.2)

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x 

