#! -*- coding:utf-8 -*-

import numpy as np

class Vocabs(object):

    def __init__(self, filename):
        #self.vocabs = '0123456789abcdefghijklmnopqrstuvwxyz .\'#$%&-+*[]<>'
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

def _random_choice(vocabs):
    return vocabs.vocabs[np.random.randint(0, len(vocabs))]

def make_train_data(vocabs, num_steps, num_data):
    ret = []
    steps = range(num_steps)
    for i in range(num_data):
        length = np.random.randint(1, num_steps + 1)
        ret.append([_random_choice(vocabs) for _ in range(length)])
    return ret

if __name__ == u'__main__':
    v = Vocabs()
    print(make_train_data(v, 10, 10))
