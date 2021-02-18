# -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf

class Trainer(object):
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam()
        
    @tf.function
    def __call__(self, model, inputs):
        with tf.GradientTape() as tape:
            loss = model.loss(inputs)

        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
