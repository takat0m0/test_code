#! -*- coding:utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F

class LSTMCell(chainer.Chain):
    def __init__(self, total_char_num, hidden_num, train = True, ratio = 0.5):
        super(LSTMCell, self).__init__(
            embed=L.EmbedID(total_char_num, hidden_num, ignore_label = -1),
            l=L.LSTM(hidden_num, hidden_num),
            output=L.Linear(hidden_num, total_char_num),
        )
        self.train = train
        self.ratio = ratio

    def __call__(self, x):
        e = self.embed(x)
        h = F.dropout(self.l(e), train = self.train, ratio = self.ratio)
        out = self.output(h)
        return out

    def set_train(self, train):
        self.train = train

    def set_ratio(self, ratio):
        self.ratio = ratio

    def get_hidden(self):
        return self.l.c, self.l.h

    def reset_state(self):
        self.l.reset_state()
