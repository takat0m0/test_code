# -*- coding:utf-8 -*-

import os
import sys

import scipy
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    num_sample = 100
    x1 = np.random.normal(0.0, 0.6, num_sample)
    x2 = np.random.normal(2.0, 0.6, 2 * num_sample)
    x = np.hstack([x1, x2])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x, bins = 50)
    plt.savefig('test.png')    

    N = len(x)
    K = 2

    model = pm.Model()
    with model:
        pi = pm.Dirichlet('pi', a = np.ones(K), shape = K)
        s  = pm.Categorical('s', p = pi, shape = N)
        mu = pm.Normal('mu', mu = 0, sd = 5.0, shape = K)
        tau = pm.Gamma('tau', alpha = 1.0, beta = 0.1, shape = K)        
        obs = pm.Normal('x', mu = mu[s], tau = tau[s], observed = x)
    with model:
        trace = pm.sample(10, tune = 200, step = pm.NUTS())
    print(trace['s'])        
    print(trace['s'].shape)
    print(trace['mu'])
    print(trace['mu'].shape)                

    for i, val in enumerate(x):
        print(val, np.mean(trace['s'][:, i]))
