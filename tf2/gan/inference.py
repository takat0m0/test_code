# -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import numpy as np
import cv2

from model import Model

if __name__ == '__main__':
    # -- parameter --
    batch_size = 32
    z_dim = 100
    
    # -- model --
    model = Model()
    model.gen = tf.keras.models.load_model('./model.dump')

    # -- make figure --
    z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)    
    figs = model.make_fig(z)

    # -- dump figs --
    figs = (figs + 1.0) * 127.5
    figs = np.reshape(figs, (batch_size, 28, 28))
    os.mkdir('dump')
    for i, fig in enumerate(figs):
        cv2.imwrite('dump/{}.png'.format(i), fig)
