#! -*- coding:utf-8 -*-
import sys
import os

import argparse
import numpy as np
import pickle

import chainer
from chainer import serializers
import chainer.links as L
import chainer.functions as F
from chainer import cuda

from Model import CNNModel
from parameters import Parameters
from read_file import get_data

if __name__ == u'__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default=None, type=str)
    parser.add_argument('--params', '-p', default=None, type=str)
    parser.add_argument('--testdata', '-t', default=None, type=str)
    args = parser.parse_args()

    if args.model is None:
        args.model = 'cnn_model.dump'
    if args.params is None:
        args.params = 'params.dump'

    # get data
    label_data, test_data = get_data(args.testdata)

    # prams
    with open(args.params, 'rb') as f:
        params = pickle.load(f)

    # cnn_model
    cnn_model = CNNModel(len(test_data[0]), len(test_data[0][0]),
                         params.filter_size, max(label_data) + 1)
    chainer.serializers.load_npz(args.model, cnn_model)


    num_miss = 0
    num = 0
    for t, l in zip(test_data, label_data):
        num += 1
        x = chainer.Variable(np.asarray([[t]], dtype = np.float32))
        data = F.softmax(cnn_model(x).data).data
        max_id = np.argmax(data)
        if l != max_id:
            print('{} : {}'.format(l, max_id))
            num_miss += 1

    print('correct_num : {}/{}, correct_rate = {}'.format(num - num_miss, num, float(num - num_miss)/num))
