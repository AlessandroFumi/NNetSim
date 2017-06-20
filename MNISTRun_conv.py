# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:55:25 2017

@author: admin-congo
"""
import numpy as np
import mnist
import NNetsim_class as nnet


full_set = [i for i in mnist.read_mnist('training')]
np.random.shuffle(full_set)
test_set = [i for i in mnist.read_mnist('testing')]