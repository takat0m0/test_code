#! -*- coding:utf-8 -*-

import chainer
import chainer.functions as F
import numpy as np

from Seq2Seq import Seq2Seq
from Serializer import Serializer

class MakeSeqFunction(object):
    def __init__(self, seq2seq, serializer, num_steps):
        self.seq2seq = seq2seq
        self.serializer = serializer
        self.num_steps = num_steps

    def _feed(self, input_words):
        for i in range(self.num_steps - len(input_words)):
            x = chainer.Variable(np.asarray([-1], dtype = np.int32))
            self.seq2seq.encode_one_step(x)
        for c in list(input_words):
            x_ = self.serializer.char2id(c)
            x = chainer.Variable(np.asarray([x_], dtype = np.int32))
            self.seq2seq.encode_one_step(x)

    def _make(self, initial):
        # greedy search

        ret = ""

        max_id = self.serializer.char2id(initial)
        for i in range(self.num_steps + 1):
            x = chainer.Variable(np.asarray([max_id], dtype = np.int32))
            data = F.softmax(self.seq2seq.decode_one_step(x).data).data
            max_id = np.argmax(data)
            ret += self.serializer.id2char(max_id)
            if max_id == self.serializer.char2id(self.serializer.EOS):
                break
        return ret

    def __call__(self, input_words):
        self.seq2seq.reset_state()
        self._feed(input_words)
        self.seq2seq.transfer()
        return self._make(self.serializer.SEP)

def make_seq(seq2seq, serializer, input_words, num_steps):
    return MakeSeqFunction(seq2seq, serializer, num_steps)(input_words)
