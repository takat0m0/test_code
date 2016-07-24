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

from Serializer import Serializer, serialize_data
from Vocabs import Vocabs, make_train_data
from Seq2Seq import Seq2Seq
from Parameters import Parameters
from MakeSeq import make_seq

if __name__ == u'__main__':

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=None, type=int)
    args = parser.parse_args()

    # calculation parameters
    parameters = Parameters()

    # get data
    vocabs = Vocabs()
    serializer = Serializer(vocabs)
    train_data = make_train_data(vocabs, 
                                 parameters.time_steps, 
                                 parameters.data_num)
    
    print('--- first 10 data ---')
    for data in train_data[0: 10]:
        print(''.join(data))

    train_data, result_data = serialize_data(serializer, train_data, 
                                             parameters.time_steps)

    # set models
    seq2seq = Seq2Seq(serializer.num_char, parameters.hidden_num)
    objective = L.Classifier(seq2seq)
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

        seq2seq.training_mode()

        tick = time.time()
        for i in range(num_one_epoch):
            seq2seq.reset_state()

            loss = 0
            s_index = i * parameters.batch_size
            t_index = (i + 1) * parameters.batch_size

            # encode phase
            for time_step in range(parameters.time_steps):
                x = array2variable(train_data[s_index:t_index, time_step])
                seq2seq.encode_one_step(x)

            # transfer
            seq2seq.transfer()
            x = array2variable([serializer.char2id(vocabs.SEP)] * (t_index - s_index))
            t = array2variable(result_data[s_index:t_index, 0])
            loss += objective(x, t)

            # decode phase
            for time_step in range(parameters.time_steps):
                x = array2variable(train_data[s_index:t_index, time_step]) 
                t = array2variable(result_data[s_index:t_index, time_step + 1])
                loss += objective(x, t)

            # optimize
            objective.zerograds()
            loss.backward()
            optimizer.update()


        tack = time.time()
    
        print("epoch:{}, loss:{}, time:{} sec".format(epoch, loss.data, tack - tick))

        # make_seq test
        seq2seq.reading_mode()
        test_input = 'test test.'
        print('input >> {}'.format(test_input))
        print('output>> {}'.format(make_seq(seq2seq, serializer, test_input, parameters.time_steps + 1)))

    print('___ end training ___');sys.stdout.flush()

    # dump
    seq2seq.to_cpu()
    chainer.serializers.save_npz('seq2seq.dump', seq2seq)

    with open('parameters.dump', 'wb') as f:
        pickle.dump(parameters, f)

    with open('serializer.dump', 'wb') as f:
        pickle.dump(serializer, f)

