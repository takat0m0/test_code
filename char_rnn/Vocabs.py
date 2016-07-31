#! -*- coding:utf-8 -*-

import numpy as np

class Vocabs(object):

    def __init__(self, filename):
        with open(filename, 'r') as f:
            tmp = list(f.read())
            tmp = set(tmp)
        self.vocabs = list(tmp)
        self.EOS = '<eos>' # end of setncence
        self.UNK = '<unk>' # unkonwn char
        self.SEP = '#####' # the sign for the start of decode

    def __len__(self):
        return len(self.vocabs)

    def __iter__(self):
        return self.vocabs.__iter__()
