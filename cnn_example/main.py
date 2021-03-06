#! -*- coding:utf-8 -*-

u'''
**** NOTICE ****
We must define a function get_data in read_file.py.
****************
'''

import sys
import os

import time
import argparse
import numpy as np
import pickle

import chainer
import chainer.links as L
from chainer import optimizers

from read_file import get_data
from parameters import Parameters
from Model import CNNModel

if __name__ == u'__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=None, type=int)
    parser.add_argument('--datafile', '-d', default=None, type=str)
    args = parser.parse_args()

    # get data
    label_data, train_data = get_data(args.datafile)

    # params
    params = Parameters()

    # set models
    cnn_model = CNNModel(len(train_data[0]), len(train_data[0][0]),
                         params.filter_size, max(label_data) + 1)
    objective = L.Classifier(cnn_model)
    objective.compute_accuracy = False

    # for cuda
    if args.gpu is None:
        xp = np
    else:
        xp = cuda.cupy
        cuda.get_device(args.gpu).use()
        objective.to_gpu()

    # set initial value
    for param in objective.params():
        data = param.data
        data[:] = np.random.uniform(-params.init_val, params.init_val, data.shape)

    # optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(objective)

    # main loop
    def array2variable(target, dtype_ = xp.int32):
        return chainer.Variable(xp.array(target,dtype = dtype_))    

    num_one_epoch = len(train_data)//params.batch_size

    print('--- begin training ---');sys.stdout.flush()
    for epoch in range(params.total_epoch):

        tick = time.time()

        for i in range(num_one_epoch):
            # calcualte loss
            loss = 0
            this_range = range(i * params.batch_size, (i + 1) * params.batch_size)
            x = array2variable([[train_data[_]] for _ in this_range], dtype_ = xp.float32) 
            t = array2variable([label_data[_] for _ in this_range])
            loss += objective(x, t)

            # optimize
            objective.zerograds()
            loss.backward()
            optimizer.update()

        tack = time.time()

        print('epoch: {}, loss = {}'.format(epoch, loss.data))

    print('___ end training ___');sys.stdout.flush()

    # dump
    cnn_model.to_cpu()
    chainer.serializers.save_npz('cnn_model.dump', cnn_model)

    with open('params.dump', 'wb') as f:
        pickle.dump(params, f)
