#! -*- coding:utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F

class CNNModel(chainer.Chain):
    def __init__(self, hight, width, filter_size, num_category):
        super(CNNModel, self).__init__(
            conv1 = L.Convolution2D(1, 1, filter_size, pad = int((filter_size - 1)/2)),
            conv2 = L.Convolution2D(1, 1, filter_size, pad = int((filter_size - 1)/2)),
            output = L.Linear(hight * width, num_category)
            )
        
    def __call__(self, input_data):
        h = F.relu(self.conv1(input_data))
        h = F.relu(self.conv2(h))
        return self.output(h)

if __name__ == u'__main__':
    c = CNNModel(28, 28, 5, 10)
    
