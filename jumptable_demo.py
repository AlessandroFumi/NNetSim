# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:52:18 2017

@author: admin-congo
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

mu1 = 30
mu2 = 70

sigma1 = 5
sigma2 = 7

x_start = 0
x_end = 100
x_granularity = 1
gauss1 = norm(loc = mu1, scale = sigma1)
gauss2 = norm(loc = mu2, scale = sigma2)
x = np.arange(x_start,x_end,x_granularity)
p_x = (2*gauss1.pdf(x) + 3*gauss2.pdf(x))/5

sumx = np.cumsum(p_x)

#plt.plot(x, sumx)

numIterations = 5000
binwidth = max(x_granularity, (x_end - x_start)/numIterations)
switchings = [np.argmax(sumx > i) for i in np.random.rand(numIterations)]
plt.figure(1, figsize = (7,7))
occurrences = plt.hist(switchings ,bins = np.arange(x_start,x_end,binwidth))
norm_occurrences = [i/(binwidth*numIterations) for i in occurrences[0]]
plt.figure(2, figsize = (7,7))
plt.bar(np.arange(x_start,x_end-binwidth,binwidth),norm_occurrences,binwidth)
plt.plot(x, p_x, color = 'r', linewidth = 3)
