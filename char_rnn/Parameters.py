#! -*- coding:utf-8 -*-

class Parameters(object):

    def __init__(self):
        self.hidden_num = 300
        self.batch_size = 200
        self.time_steps = 50
        self.total_epoch = 200

        self.init_val = 0.2

    def printing(self):
        print('* hidden      = {}'.format(self.hidden_num))
        print('* batch size  = {}'.format(self.batch_size))
        print('* time step   = {}'.format(self.time_steps))
        print('* total epoch = {}'.format(self.total_epoch))

