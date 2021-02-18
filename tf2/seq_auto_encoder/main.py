# -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import numpy as np

from trainer import Trainer
from model import Model

if __name__ == '__main__':
    # -- parameter --
    batch_size = 32
    input_dim = 5
    h_dim = 64

    train_num = 1000
    time_length = 10
    
    num_epoch = 500
    
    # -- data --
    train_data = np.random.uniform(-1, 1, (train_num, time_length, input_dim))

    # -- trainer and model --
    trainer = Trainer()
    model = Model(batch_size, input_dim, h_dim)

    # -- train loop --
    num_one_epoch = train_num // batch_size
    idxs = [_ for _ in range(train_num)]
    
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(encoder = model.enc,
                                     decoder = model.dec)
    
    for epoch in range(num_epoch):
        # -- shuffle data --
        tmp_idxs = np.random.permutation(idxs)
        train_data = train_data[tmp_idxs]

        total_loss = 0.0
        
        for i in range(num_one_epoch):
            batch_data = train_data[i * batch_size: (i + 1) * batch_size]
            loss = trainer(model, batch_data)

            loss = loss.numpy()            
            total_loss += loss/num_one_epoch

        print('epoch:{}, loss:{}'.format(epoch, total_loss))

        checkpoint.save(file_prefix = checkpoint_prefix)
