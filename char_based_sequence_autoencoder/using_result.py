#! -*- coding:utf-8 -*-
import sys
import os

import time
import argparse
import numpy as np
import pickle

import chainer
from chainer import serializers
import chainer.links as L
from chainer import cuda

from Serializer import Serializer, get_data
from Seq2Seq import Seq2Seq
from Parameters import Parameters
from MakeSeq import make_seq

if __name__ == u'__main__':

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq2seq', '-s2s', default=None, type=str)
    parser.add_argument('--serializer', '-s', default=None, type=str)
    parser.add_argument('--parameters', '-p', default=None, type=str)
    args = parser.parse_args()
    if args.seq2seq is None:
        args.seq2seq = 'seq2seq.dump'
    if args.serializer is None:
        args.serializer = 'serializer.dump'
    if args.parameters is None:
        args.parameters = 'parameters.dump'

    # load
    with open(args.serializer, 'rb') as f:
        serializer = pickle.load(f)

    with open(args.parameters, 'rb') as f:
        parameters = pickle.load(f)

    seq2seq = Seq2Seq(serializer.num_char, parameters.hidden_num, train = False)
    chainer.serializers.load_npz(args.seq2seq, seq2seq)

    while(1):
        print('\nplease input 10 chars')
        print('    0123456789')
        input_chars = input('>>> ')
        if input_chars == 'exit' or input_chars == 'q':
            break
        if input_chars == '':
            continue
        print(make_seq(seq2seq, serializer, input_chars, parameters.time_steps + 1))
