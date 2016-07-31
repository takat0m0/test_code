#! -*- coding:utf-8 -*-

import sys
import os

import time
import argparse
import numpy as np
import pickle

import chainer

import chainer.links as L
from chainer import cuda

from Serializer import Serializer, get_data
from Vocabs import Vocabs, make_train_data
from Cell import LSTMCell
from Parameters import Parameters
from MakeSeq import make_seq

if __name__ == u'__main__':

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=None, type=int)
    parser.add_argument('--datafile', '-d', default=None, type=str)
    args = parser.parse_args()

    # calculation parameters
    parameters = Parameters()

    # get data
    vocabs = Vocabs(args.datafile)
    serializer = Serializer(vocabs)
    train_data, result_data = get_data(args.datafile, 
                                       serializer, 
                                       parameters.time_steps)


    # set models
    cell = LSTMCell(serializer.num_char, parameters.hidden_num)
    objective = L.Classifier(cell)
    objective.compute_accuracy = False

    # for cuda
    if args.gpu is None:
        xp = np
    else:
        xp = cuda.cupy
        cuda.get_device(args.gpu).use()
        aobjective.to_gpu()

    # set initial value
    for param in objective.params():
        data = param.data
        data[:] = np.random.uniform(-parameters.init_val, parameters.init_val, data.shape)


    # optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(objective)

    # main loop
    def array2variable(target, dtype_ = xp.int32):
        return chainer.Variable(xp.array(target,dtype = dtype_))    

    print('---   parameters   ---');sys.stdout.flush()
    parameters.printing()

    print('--- begin training ---');sys.stdout.flush()
    num_one_epoch = len(train_data)//parameters.batch_size

    for epoch in range(parameters.total_epoch):

        cell.set_train(True)

        tick = time.time()
        for i in range(num_one_epoch):
            cell.reset_state()

            loss = 0
            s_index = i * parameters.batch_size
            t_index = (i + 1) * parameters.batch_size

            # decode phase
            for time_step in range(parameters.time_steps):
                x = array2variable(train_data[s_index:t_index, time_step]) 
                t = array2variable(result_data[s_index:t_index, time_step])
                loss += objective(x, t)

            # optimize
            objective.zerograds()
            loss.backward()
            optimizer.update()


        tack = time.time()
    
        print("epoch:{}, loss:{}, time:{} sec".format(epoch, loss.data, tack - tick))

        # make_seq test
        cell.set_train(False)
        test_input = 'the '
        print(make_seq(cell, serializer, test_input, parameters.time_steps))

    print('___ end training ___');sys.stdout.flush()

    # dump
    cell.to_cpu()
    chainer.serializers.save_npz('cell.dump', cell)

    with open('parameters.dump', 'wb') as f:
        pickle.dump(parameters, f)

    with open('serializer.dump', 'wb') as f:
        pickle.dump(serializer, f)

