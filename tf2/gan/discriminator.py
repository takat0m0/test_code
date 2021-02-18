# -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import numpy as np

class Discriminator(tf.keras.Model):
    def  __init__(self):
        super(Discriminator, self).__init__()
        self.conv0 = tf.keras.layers.Conv2D(64, 5, strides = (2, 2),
                                           padding = 'same', name = 'conv0')
        self.lrelu0 = tf.keras.layers.LeakyReLU()

        self.conv1 = tf.keras.layers.Conv2D(128, 5, strides = (2, 2),
                                           padding = 'same', name = 'conv1')
        self.lrelu1 = tf.keras.layers.LeakyReLU()        
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense0 = tf.keras.layers.Dense(1, name = 'dense0')

    @tf.function
    def call(self, inputs):
        h = self.lrelu0(self.conv0(inputs))
        h = self.lrelu1(self.conv1(h))
        h = self.flatten(h)
        logit = self.dense0(h)
        return logit
    
if __name__ == '__main__':
    shape = (5, 28, 28, 1)
    inputs = tf.random.normal(shape = shape)
    discriminator = Discriminator()
    discriminator(inputs)
