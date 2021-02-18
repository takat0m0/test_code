# -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import numpy as np

from generator import Generator
from discriminator import Discriminator

class Model(object):
    def __init__(self):
        self.gen  = Generator()
        self.disc = Discriminator()        

    def make_fig(self, z):
        return self.gen(z, training = False)

    
    @tf.function
    def gen_loss(self, z):
        figs = self.gen(z)
        logits = self.disc(figs)
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = logits, labels = tf.ones_like(logits))
            )
        return loss
    
    @tf.function
    def disc_loss(self, figs, z):
        fake_figs = self.gen(z)
        g_logits = self.disc(fake_figs)
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = g_logits, labels = tf.zeros_like(g_logits))
            )
        
        d_logits = self.disc(figs)
        d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = d_logits, labels = tf.ones_like(d_logits))
            )
        
        return g_loss + d_loss
    
    @property
    def gen_trainable_variables(self):
        return self.gen.trainable_variables
    
    @property
    def disc_trainable_variables(self):
        return self.disc.trainable_variables
    
    
if __name__ == '__main__':
    model = Model()

