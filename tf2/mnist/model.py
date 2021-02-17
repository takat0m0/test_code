# -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import numpy as np

class Network(tf.keras.Model):
    def  __init__(self, num_output):
        super(Network, self).__init__()
        self.conv0 = tf.keras.layers.Conv2D(5, 3, activation = 'relu',
                                           padding = 'same', name = 'conv0')
        self.conv1 = tf.keras.layers.Conv2D(10, 3, activation = 'relu',
                                           padding = 'same', name = 'conv1')
        self.conv2 = tf.keras.layers.Conv2D(10, 3, activation = 'relu',
                                           padding = 'same', name = 'conv2')        
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense0 = tf.keras.layers.Dense(num_output, name = 'dense0')
        
    @tf.function
    def call(self, inputs):
        h = self.conv0(inputs)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.flatten(h)
        logit = self.dense0(h)
        return logit

class Model(object):
    def __init__(self, num_output):
        self.network = Network(num_output)

    def get_logits(self, inputs):
        return self.network(inputs)

    def get_probs(self, inputs):
        return tf.nn.softmax(self.network(inputs))
    
    @tf.function
    def loss(self, inputs, labels):
        logits = self.network(inputs)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits = logits, labels = labels)
            )
        return loss
        
    @property
    def trainable_variables(self):
        return self.network.trainable_variables

    def __call__(self, inputs):
        return self.network(inputs)
    
if __name__ == '__main__':
    shape = (5, 28, 28, 1)
    inputs = tf.random.normal(shape = shape)
    model = Model(10)
    logits = model(inputs)
    probs = tf.nn.softmax(logits)
    print(model.trainable_variables)
