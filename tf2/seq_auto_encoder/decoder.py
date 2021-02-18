# -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import numpy as np

class Decoder(tf.keras.Model):
    def  __init__(self, batch_size, h_dim, input_dim):
        super(Decoder, self).__init__()
        
        self.batch_size = batch_size
        self.h_dim = h_dim
        self.input_dim = input_dim
        
        self.dense0 = tf.keras.layers.Dense(24,
                                            name = 'dec_dense0',
                                            activation = 'relu')
        self.lstm = tf.keras.layers.LSTMCell(h_dim, name = 'dec_lstm')
        self.dense1 = tf.keras.layers.Dense(input_dim, name = 'dec_dense1')
        self.set_initial()
        
    def _make_initial_state(self):
        return [tf.zeros((self.batch_size, self.h_dim)),
                tf.zeros((self.batch_size, self.h_dim))]
    
    def _make_eos_state(self):
        return tf.ones((self.batch_size, self.input_dim))
    
    def set_initial(self):
        self.previouse_state = self._make_initial_state()
        
    def set_previouse(self, states):
        self.previouse_state = states
        
    def call(self, inputs):
        # inputs = [batch, feature]
        h = self.dense0(inputs)
        h, self.previouse_state = self.lstm(h, self.previouse_state)
        h = self.dense1(h)
        return h

    def decoding(self, enc_states, length):

        self.set_previouse(enc_states)
        
        ret = []
        inputs = self._make_eos_state()
        
        for t in range(length):
            h = self(inputs)
            inputs = h
            ret.append(h)
        return ret
    
if __name__ == '__main__':
    batch_size = 32
    input_dim = 5
    time_length = 10
    shape = (batch_size, time_length, input_dim)
    
    h_dim = 64    

    inputs = np.random.uniform(-1, 1, shape)
    dec = Decoder(batch_size, h_dim, input_dim)
    
    ret = dec.decoding([tf.zeros((batch_size, h_dim)),
                        tf.zeros((batch_size, h_dim))], time_length)
    print(ret)
    print(dec.trainable_variables)
