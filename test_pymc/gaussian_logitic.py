# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import theano.tensor as tt

class Kernel(object):
    def __init__(self, xs, ell, factor, sigma):
        self.xs = xs
        self.ell = ell
        self.factor = factor * factor
        self.sigma = sigma
        self._get_matrix()
        
    def _basic_func(self, x, x_):
        norm = np.linalg.norm(x - x_)
        return self.factor * np.exp(-np.square(norm)/(2 * self.ell * self.ell))
    
    def _get_matrix(self):
        self.K_matrix = np.zeros((len(self.xs), len(self.xs)))
        for i, x in enumerate(self.xs):
            for j, x_ in enumerate(self.xs):
                self.K_matrix[i, j] = self._basic_func(x, x_)
        self.K_matrix = self.K_matrix + self.sigma * np.eye(len(self.xs))
        self.K_inv_matrix = np.linalg.inv(self.K_matrix)
        
    def get_k_vector(self, x):
        ret = np.zeros(len(self.xs))
        for i, x_ in enumerate(self.xs):
            ret[i] = self._basic_func(x, x_)
        return ret

    def get_k_sigma(self, x):
        return self.sigma + self._basic_func(x, x)


def get_variable_for_predict(kernel, x, ys):
    k_vec = kernel.get_k_vector(x)
    k_sigma = kernel.get_k_sigma(x)

    mu = np.dot(k_vec, np.dot(kernel.K_inv_matrix, ys))
    sigma = k_sigma - np.dot(k_vec, np.dot(kernel.K_inv_matrix, k_vec))

    return mu, sigma

def make_output(x):
    norm = np.linalg.norm(x)
    if norm > 1.0:
        return 1.0
    else:
        return 0.0
    
if __name__ == '__main__':
    data_num = 100
    input_dim = 2
    
    x = np.random.uniform(0.0, 1.0, (data_num, input_dim))
    y = np.array([make_output(_) for _ in x])

    model = pm.Model()
    with model:
        ell    = pm.HalfNormal('ell', sd = 2.0)
        factor = pm.HalfNormal('factor', sd = 2.0)
        nu     = pm.HalfNormal('nu', sd = 1.0)        
        cov = factor ** 2 * pm.gp.cov.ExpQuad(input_dim, ell)
        total_sigma = nu * np.eye(data_num) + cov(x.reshape(-1, input_dim))
        
        f = pm.MvNormal('f', mu = np.zeros(data_num), cov = total_sigma, shape = data_num)
        prob = pm.math.sigmoid(f)
        y_ = pm.Bernoulli('y', prob, observed = y)
        
    with model:
        trace = pm.sample(100, tune = 3000, step = pm.NUTS())

    ell_mean = np.mean(trace['ell'])
    factor_mean = np.mean(trace['factor'])
    sigma_mean = np.mean(trace['nu'])
    f_mean = np.mean(trace['f'], axis = 0)

    predict_num = 200
    K = Kernel(x, ell_mean, factor_mean, sigma_mean)
    predict_x = np.random.uniform(0, 1, (predict_num, input_dim))
    predict_logits = [get_variable_for_predict(K, _, f_mean)[0] for _ in predict_x]

    predicts = [1.0/(1.0 + np.exp(-logit)) for logit in predict_logits]
    
    for i in range(predict_num):
        gt = make_output(predict_x[i])
        norm = np.linalg.norm(predict_x[i])
        print('{}: gt = {}, predict = {}'.format(norm,
                                                 gt,
                                                 predicts[i]))
