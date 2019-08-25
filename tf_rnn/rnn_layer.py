# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from tf_util import Layers

class RNNLayers(Layers):
    def __init__(self, name_scopes, hidden_dims):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self.hidden_dims = hidden_dims    

    def set_model(self, inputs, sequence_length, is_tarining = True, reuse = False):
        '''
        inputs: [batch, max_time, data_dim]
        '''
        h = inputs
        batch_size = tf.shape(inputs)[0]
        with tf.variable_scope(self.name_scopes[0], reuse = reuse):
            for i, n_hidden in enumerate(self.hidden_dims):
                with tf.variable_scope('RNN_{}'.format(i), reuse = reuse):
                    cell = tf.nn.rnn_cell.GRUCell(n_hidden)
                    initial_state = cell.zero_state(batch_size, dtype=tf.float32)
                    h, state = tf.nn.dynamic_rnn(
                        cell, h, initial_state=initial_state,
                        sequence_length=sequence_length)
        return state

if __name__ == '__main__':
    max_time_length = 5
    data_dim = 30
    rnn = RNNLayers(['RNNLAYERS'], [10, 10])
    inputs = tf.placeholder(dtype = tf.float32,
                            shape = [None, max_time_length, data_dim])
    sequence_length = tf.placeholder(dtype = tf.int32,
                                     shape = [None])    
    outputs = rnn.set_model(inputs, sequence_length,
                            True, False)
    print(outputs)
    
