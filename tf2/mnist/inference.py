# -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import numpy as np

from mnist_download import get_data
from model import Model

if __name__ == '__main__':
    # -- parameter --
    batch_size = 256
    num_epoch = 10
    
    # -- data --
    train_num = 50000
    imgs, labels = get_data()
    test_imgs, test_labels   = imgs[train_num:], labels[train_num:]

    # -- model --
    model = Model(len(test_labels[0]))
    model.network = tf.keras.models.load_model('./model.dump')

    for img, label in zip(test_imgs[:100], test_labels[:100]):
        print(model.get_probs([img]).numpy()[0])
        print(label)
