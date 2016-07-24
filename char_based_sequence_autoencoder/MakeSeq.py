#! -*- coding:utf-8 -*-

import chainer
import chainer.functions as F
import numpy as np

from Seq2Seq import Seq2Seq
from Serializer import Serializer

class MakeSeqFunction(object):
    def __init__(self, seq2seq, serializer):
        self.seq2seq = seq2seq
        self.serializer = serializer

    def _feed(self, input_words):
        for c in list(input_words):
            x_ = self.serializer.char2id(c)
            x = chainer.Variable(np.asarray([x_], dtype = np.int32))
            self.seq2seq.encode_one_step(x)

    def _make(self, initial, char_num):
        # greedy search

        ret = ""

        max_id = self.serializer.char2id(initial)
        for i in range(char_num):
            x = chainer.Variable(np.asarray([max_id], dtype = np.int32))
            data = F.softmax(self.seq2seq.decode_one_step(x).data).data
            max_id = np.argmax(data)
            ret += self.serializer.id2char(max_id)

        return ret

    def __call__(self, input_words, char_num):
        self.seq2seq.reset_state()
        self._feed(input_words)
        self.seq2seq.transfer()
        return self._make(self.serializer.SEP, char_num)

def make_seq(seq2seq, serializer, input_words, char_num):
    return MakeSeqFunction(seq2seq, serializer)(input_words, char_num)
