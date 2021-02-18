# -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf

class Trainer(object):
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam(lr = 1.0e-4, beta_1 = 0.5)
        
    @tf.function
    def __call__(self, model, figs, z):
        with tf.GradientTape() as tape:
            g_loss = model.gen_loss(z)
        grads = tape.gradient(g_loss, model.gen_trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.gen_trainable_variables))
        with tf.GradientTape() as tape:
            d_loss = model.disc_loss(figs, z)
        grads = tape.gradient(d_loss, model.disc_trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.disc_trainable_variables))        
        
        return g_loss, d_loss
    
if __name__ == '__main__':
    trainer = Trainer()
