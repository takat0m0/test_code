#! -*- coding:utf-8 -*-

class Parameters(object):
    def __init__(self):
        self.data_num = 20000

        self.hidden_num = 200
        self.batch_size = 100
        self.time_steps = 10
        self.total_epoch = 100

        self.init_val = 0.2

    def printing(self):
        print('* hidden      = {}'.format(self.hidden_num))
        print('* batch size  = {}'.format(self.batch_size))
        print('* time step   = {}'.format(self.time_steps))
        print('* total epoch = {}'.format(self.total_epoch))

