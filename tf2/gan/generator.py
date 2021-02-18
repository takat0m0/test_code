# -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import numpy as np

class Generator(tf.keras.Model):
    def  __init__(self):
        super(Generator, self).__init__()
        self.dense0 = tf.keras.layers.Dense(7 * 7 * 256, use_bias = False, name = 'dense0')
        self.bn0    = tf.keras.layers.BatchNormalization(name = 'bn_dense0')
        self.lrelu0 = tf.keras.layers.LeakyReLU()

        self.reshape = tf.keras.layers.Reshape((7, 7, 256))

        self.deconv1 = tf.keras.layers.Conv2DTranspose(128, 5, padding = 'same', use_bias = False, name = 'deconv1')
        self.bn1     = tf.keras.layers.BatchNormalization(name = 'bn_deconv1')
        self.lrelu1  = tf.keras.layers.LeakyReLU()
        
        self.deconv2 = tf.keras.layers.Conv2DTranspose(64, 5, strides = (2, 2), padding = 'same', use_bias = False, name = 'deconv2')
        self.bn2     = tf.keras.layers.BatchNormalization(name = 'bn_deconv2')
        self.lrelu2  = tf.keras.layers.LeakyReLU()
        
        self.deconv3 = tf.keras.layers.Conv2DTranspose(1, 5, strides = (2, 2), padding = 'same', use_bias = False, name = 'deconv3')
        
    @tf.function
    def call(self, inputs, training = True):
        h = self.lrelu0(self.bn0(self.dense0(inputs), training = training))
        
        h = self.reshape(h)
        
        h = self.lrelu1(self.bn1(self.deconv1(h), training = training))

        h = self.lrelu2(self.bn2(self.deconv2(h), training = training))
        
        fig = tf.tanh(self.deconv3(h))
        
        return fig
    
if __name__ == '__main__':
    shape = (5, 100)
    inputs = tf.random.normal(shape = shape)
    generator = Generator()
    figs = generator(inputs)
