# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from tf_util import get_rand_variable, get_const_variable, lrelu, linear, Layers

class FeatureExtractor(Layers):
    def __init__(self, name_scopes, layer_dims):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self.layer_dims = layer_dims
        
    def set_model(self, inputs, is_training = True, reuse = False):
        '''
        inputs: [batch, max_time, data_dim]
        '''
        in_dim = inputs.get_shape()[-1]
        h = inputs
        with tf.variable_scope(self.name_scopes[0], reuse = reuse):
            for i, out_dim in enumerate(self.layer_dims):
                w = get_rand_variable('weight_{}'.format(i),
                                      [in_dim, out_dim],
                                      0.02)
                b = get_rand_variable('bias_{}'.format(i),
                                      [out_dim],
                                      0.0)
                lin = tf.einsum('ij,kli->klj', w, h) + b
                h = lrelu(lin)
                in_dim = out_dim
        return h
    
class LinearLayers(Layers):
    def __init__(self, name_scopes, layer_dims):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self.layer_dims = layer_dims
        
    def set_model(self, inputs, is_training = True, reuse = False):
        '''
        inputs: [batch, data_dim]
        '''
        h = inputs
        with tf.variable_scope(self.name_scopes[0], reuse = reuse):
            for i, out_dim in enumerate(self.layer_dims):
                lin = linear('LL_{}'.format(i), h, out_dim)
                h = lrelu(lin)
        return lin

if __name__ == '__main__':
    max_time_length = 5
    data_dim = 30
    fe = FeatureExtractor(['FEATURE'], [15, 15, 10])
    inputs = tf.placeholder(dtype = tf.float32,
                            shape = [None, max_time_length, data_dim])
    outputs = fe.set_model(inputs, True, False)
    print(outputs)
    
    ll = LinearLayers(['LINEARLAYER'], [20, 15, 15])
    inputs = tf.placeholder(dtype = tf.float32,
                            shape = [None, data_dim])
    outputs = ll.set_model(inputs, True, False)
    print(outputs)
