# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from linear_layer import FeatureExtractor, LinearLayers
from rnn_layer import RNNLayers

class Model(object):
    def __init__(self, max_time_length, data_dim):
        self.max_time_length = max_time_length
        self.data_dim = data_dim
        self.fe = FeatureExtractor(['FEATURE'], [20, 20, 15])
        self.rnn = RNNLayers(['RNNLAYERS'], [10, 20])
        self.ll = LinearLayers(['LINEARLAYER'], [10, 5])

    def set_model(self):
        # -- place holder --
        self.inputs = tf.placeholder(dtype = tf.float32,
                                     shape = [None, self.max_time_length, self.data_dim])
        self.sequence_length = tf.placeholder(dtype = tf.int32, shape = [None])
        
        # -- make networks --
        fv = self.fe.set_model(self.inputs, True, False)
        rnn_fv = self.rnn.set_model(fv, self.sequence_length, True, False)
        outputs = self.ll.set_model(rnn_fv, True, False)
        
        # -- for inference --
        fv = self.fe.set_model(self.inputs, False, True)
        rnn_fv = self.rnn.set_model(fv, self.sequence_length, False, True)
        self.outputs_wo_train = self.ll.set_model(rnn_fv, False, True)
        print(self.outputs_wo_train)
    def get_outputs(sess, inputs, sequence_length):
        feed_dict = {self.inputs: inputs,
                     self.sequence_length: sequence_length}
        ret = sess.run(self.outputs_wo_train, feed_dict = feed_dict)
        return ret
    
if __name__ == '__main__':
    model = Model(max_time_length = 8, data_dim = 30)
    model.set_model()
