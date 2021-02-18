# -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import numpy as np

class Encoder(tf.keras.Model):
    def  __init__(self, batch_size, h_dim):
        super(Encoder, self).__init__()
        
        self.batch_size = batch_size
        self.h_dim = h_dim
        
        self.dense0 = tf.keras.layers.Dense(24,
                                            name = 'enc_dense0',
                                            activation = 'relu')
        self.lstm = tf.keras.layers.LSTMCell(h_dim, name = 'enc_lstm')
        self.set_initial()
        
    def _make_initial_state(self):
        return [tf.zeros((self.batch_size, self.h_dim)),
                tf.zeros((self.batch_size, self.h_dim))]
    
    def set_initial(self):
        self.previouse_state = self._make_initial_state()
        
    def call(self, inputs):
        # inputs = [batch, feature]
        h = self.dense0(inputs)
        h, self.previouse_state = self.lstm(h, self.previouse_state)
        return h

    def encoding(self, inputs):
        # inpouts = [batch, timestep, feature]
        #self.set_initial()
        ret = []
        timesteps = len(inputs[0])
        for t in range(timesteps):
            h = self(inputs[:, t, :])
            ret.append(h)
        return ret, self.previouse_state

    
if __name__ == '__main__':
    batch_size = 32
    h_dim = 64
    shape = (batch_size, 10, 5)
    inputs = np.random.uniform(-1, 1, shape)
    enc = Encoder(batch_size, h_dim)
    ret = enc.encoding(inputs)
    print(ret)
    print(enc.trainable_variables)
