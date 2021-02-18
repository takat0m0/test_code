# -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import numpy as np
import cv2

from mnist_download import get_data
from trainer import Trainer
from model import Model

if __name__ == '__main__':
    # -- parameter --
    z_dim = 100    
    batch_size = 256
    num_epoch = 100
    
    # -- data --
    train_num = 10000
    imgs, _ = get_data()
    train_imgs = imgs[:train_num]

    # -- trainer and model --
    trainer = Trainer()
    model = Model()

    # -- train loop --
    num_one_epoch = train_num // batch_size
    idxs = [_ for _ in range(train_num)]
    
    for epoch in range(num_epoch):
        # -- shuffle data --
        tmp_idxs = np.random.permutation(idxs)
        train_imgs = train_imgs[tmp_idxs]

        g_total_loss = 0.0
        d_total_loss = 0.0
        
        for i in range(num_one_epoch):
            batch_imgs = train_imgs[i * batch_size: (i + 1) * batch_size]
            z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            g_loss, d_loss = trainer(model, batch_imgs, z)

            g_loss = g_loss.numpy()
            g_total_loss += g_loss/num_one_epoch
            
            d_loss = d_loss.numpy()            
            d_total_loss += d_loss/num_one_epoch

        print('epoch:{}, g_loss:{}, d_loss:{}'.format(epoch, g_total_loss, d_total_loss))
        model.gen.save('./model.dump')
        z = np.random.uniform(-1, 1, [1, z_dim]).astype(np.float32)
        fig = model.make_fig(z)[0]
        fig = (fig + 1.0) * 127.5
        fig = np.reshape(fig, (28, 28))
        cv2.imwrite('test.png', fig)
