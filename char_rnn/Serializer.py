#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np

from Vocabs import Vocabs 


class Serializer(object):
    def __init__(self, vocabs):
        self.__char2id = {}
        self.__id2char = []
        for i, c in enumerate(vocabs):
            self.__char2id[c] = i
            self.__id2char.append(c)
        self._add_char(vocabs.EOS)
        self._add_char(vocabs.SEP)
        self._add_char(vocabs.UNK)

        self.EOS = vocabs.EOS
        self.SEP = vocabs.SEP
        self.UNK = vocabs.UNK

    def _add_char(self, c):
        if c in self.__char2id:
            return

        length = len(self.__char2id)
        self.__char2id[c] = length
        self.__id2char.append(c)

    def char2id(self, c):
        try:
            ret = self.__char2id[c]
            return ret
        except:
            return self.__char2id[self.UNK]

    def id2char(self, id):
        try:
            ret = self.__id2char[id]
            return ret
        except:
            return self.__id2char[self.UNK]

    @property
    def num_char(self):
        return len(self.__id2char)

def _file2charlist(filename):
    with open(filename) as f:
        tmp = f.read()
    return list(tmp)

def get_data(filename, serializer, time_steps):
    ret_x = []
    ret_t = []

    data = _file2charlist(filename)

    num_total_batch = len(data)//time_steps
    for i in range(num_total_batch):
        ret_x.append([serializer.char2id(_) for _ in data[i * time_steps: (i + 1) * time_steps]])
        ret_t.append([serializer.char2id(_) for _ in data[i * time_steps + 1: (i + 1) * time_steps + 1]])

    return np.array(ret_x, dtype = np.int32),\
        np.array(ret_t, dtype = np.int32)

if __name__ == u'__main__':
    v = Vocabs()
    s = Serializer(v)
    print(s.num_char)


