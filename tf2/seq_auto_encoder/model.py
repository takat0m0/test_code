# -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import numpy as np

from encoder import Encoder
from decoder import Decoder

class Model(object):
    def __init__(self, batch_size, input_dim, h_dim):
        
        self.enc = Encoder(batch_size, h_dim)
        self.dec = Decoder(batch_size, h_dim, input_dim)

    def loss(self, inputs):
        # inputs: [batch, timelength, feature]
        transposed = tf.transpose(inputs, (1, 0, 2))
        
        self.enc.set_initial()
        _, enc_states = self.enc.encoding(inputs)
        decodings = self.dec.decoding(enc_states, len(inputs[0]))

        loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(transposed - decodings), axis = (0, 2))
        )
        return loss
    
    def auto_encoding(self, inputs):
        self.enc.set_initial()
        _, enc_states = self.enc.encoding(inputs)
        decodings = self.dec.decoding(enc_states, len(inputs[0]))
        return decodings
    
    @property
    def trainable_variables(self):
        ret = self.enc.trainable_variables
        ret.extend(self.dec.trainable_variables)
        return ret
    
if __name__ == '__main__':
    model = Model()

