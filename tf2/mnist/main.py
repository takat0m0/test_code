# -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import numpy as np

from mnist_download import get_data
from trainer import Trainer
from model import Model

if __name__ == '__main__':
    # -- parameter --
    batch_size = 256
    num_epoch = 10
    
    # -- data --
    train_num = 50000
    imgs, labels = get_data()
    train_imgs, train_labels = imgs[:train_num], labels[:train_num]
    test_imgs, test_labels   = imgs[train_num:], labels[train_num:]

    # -- trainer and model --
    trainer = Trainer()
    model = Model(len(train_labels[0]))

    # -- train loop --
    num_one_epoch = train_num // batch_size
    idxs = [_ for _ in range(train_num)]
    
    for epoch in range(num_epoch):
        # -- shuffle data --
        tmp_idxs = np.random.permutation(idxs)
        train_imgs   = train_imgs[tmp_idxs]
        train_labels = train_labels[tmp_idxs]

        total_loss = 0.0
        for i in range(num_one_epoch):
            batch_imgs = train_imgs[i * batch_size: (i + 1) * batch_size]
            batch_labels = train_labels[i * batch_size: (i + 1) * batch_size]
            loss = trainer(model, batch_imgs, batch_labels)
            loss = loss.numpy()
            total_loss += loss/num_one_epoch
        print('epoch:{}, loss:{}'.format(epoch, total_loss))
        model.network.save('./model.dump')
