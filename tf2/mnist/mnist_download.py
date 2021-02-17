# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

def make_labels(target):
    ret = []
    for l in target:
        tmp = [0.0] * 10
        tmp[l] = 1.0
        ret.append(tmp)
    return np.asarray(ret, dtype = np.float32)

def rescale(imgs):
    imgs = imgs/127.5
    imgs = imgs - 1.0
    return imgs

def get_data():
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    shape = x.shape
    shape = [shape[0], shape[1], shape[2], 1]
    x = np.reshape(x, shape)
    x = rescale(x)

    y = make_labels(y)
    return x, y

if __name__ == '__main__':
    x, y = get_data()
    print(x[0])
