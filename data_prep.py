# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:53:51 2017

@author: admin-congo
"""
import numpy as np
import mnist
import NNvars


full_dataset = [i for i in mnist.read_mnist("training")]

np.random.shuffle(full_dataset)
training_set = full_dataset[:NNvars.numSamples]
verification_set = full_dataset[NNvars.numSamples:NNvars.numVerification+NNvars.numSamples]

test_set = [i for i in mnist.read_mnist("testing")]