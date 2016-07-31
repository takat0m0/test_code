#! -*- coding:utf-8 -*-

import chainer
import chainer.functions as F
import numpy as np

from Cell import LSTMCell
from Serializer import Serializer

class MakeSeqFunction(object):
    def __init__(self, cell, serializer):
        self.cell = cell
        self.serializer = serializer

    def _feed(self, input_words):
        for c in list(input_words)[:-1]:
            x_ = self.serializer.char2id(c)
            x = chainer.Variable(np.asarray([x_], dtype = np.int32))
            self.cell(x)

    def _make(self, initial, num_steps):
        # greedy search

        ret = ""

        max_id = self.serializer.char2id(initial)
        for i in range(num_steps):
            x = chainer.Variable(np.asarray([max_id], dtype = np.int32))
            data = F.softmax(self.cell(x).data).data
            max_id = np.argmax(data)
            ret += self.serializer.id2char(max_id)
        return ret

    def __call__(self, input_words, num_steps):
        self.cell.reset_state()
        self._feed(input_words)
        return input_words + self._make(input_words[-1], num_steps)

def make_seq(cell, serializer, input_words, num_steps):
    return MakeSeqFunction(cell, serializer)(input_words, num_steps)
