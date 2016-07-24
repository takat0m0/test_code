#! -*- coding:utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F

from Cell import EncodeCell, DecodeCell

class Seq2Seq(chainer.Chain):
    def __init__(self, total_char_num, hidden_num, train = True, ratio = 0.5):
        super(Seq2Seq, self).__init__(
            encoder = EncodeCell(total_char_num, hidden_num, train, ratio),
            decoder = DecodeCell(total_char_num, hidden_num, train, ratio),
            #translate = L.linear(hidden_num, hidden_num)
    )
    def training_mode(self, ratio = 0.5):
        self.encoder.set_train(True)
        self.decoder.set_train(True)

        self.encoder.set_ratio(ratio)
        self.decoder.set_ratio(ratio)

    def reading_mode(self):
        self.encoder.set_train(False)
        self.decoder.set_train(False)

    def transfer(self):
        c, h = self.encoder.get_hidden()
        # c = translate(c); h = translate(h)
        self.decoder.set_hidden(c, h)

    def encode_one_step(self, x):
        self.encoder(x)

    def decode_one_step(self, x):
        return self.decoder(x)

    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()

    def __call__(self, x):
        return self.decode_one_step(x)

if __name__ == u'__main__':
    s = Seq2Seq(5, 10)
