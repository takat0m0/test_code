# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    num_sample = 100
    x = np.random.uniform(-10.0, 10, num_sample)
    y = 3.5 * x + 2.2 + np.random.uniform(-5.0, 5.0, num_sample)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y)
    plt.savefig('test.png')    

    model = pm.Model()

    with model:
        a = pm.Normal('a', mu = 0, sd = 5.0)
        b = pm.Normal('b', mu = 0, sd = 5.0)
        sigma = pm.HalfNormal('sigma', sd = 1.0)        
        mu = a * x + b
        obs = pm.Normal('y', mu = mu, sd = sigma, observed = y)
        
    with model:
        trace = pm.sample(10, tune = 100, step = pm.NUTS())
    print(trace['a'])

    inputs = np.asarray([0.1 * i - 20 for i in range(400)])
    lower = []
    upper = []
    for x in inputs:
        tmp = []
        for i in range(10):
            a = trace['a'][i]
            b = trace['b'][i]
            sigma = trace['sigma'][i]        
            tmp.append(np.random.normal(a * x + b, sigma, 100))
        tmp = np.asarray(tmp)
        mean, var = np.mean(tmp), np.var(tmp)
        lower.append(mean - 3 * var)
        upper.append(mean + 3 * var)
        
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.fill_between(inputs, upper, lower, alpha = 0.5)
    plt.savefig('variance.png')    

    
        
