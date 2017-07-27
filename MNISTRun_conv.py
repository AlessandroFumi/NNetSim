# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:55:25 2017

@author: admin-congo
"""
import numpy as np
import NNetSim as nnet
import mnist

full_set = [i for i in mnist.read_mnist('training')]
np.random.shuffle(full_set)
full_set = full_set[:5000]
numEpochs = 10
# Extracting labels and images, assuming that the training set is a list of tuples

test_set = [i for i in mnist.read_mnist('testing')]

"""
This creates an empty NeuralNet object
"""
x = nnet.NeuralNet()
"""
We add the layers one by one
"""
x.add_layer(nnet.norm_layer, X_shape = (10,784))
x.add_layer(nnet.fc_layer,  out_shape = (10), learning_rate = 10)
x.add_layer(nnet.sigmoid_layer, slope = 0.1)

for i in range(numEpochs):
    print('Training Epoch {}:'.format(int(i+1)))
    x.train(full_set)
    np.random.shuffle(full_set)
    print()

for i in x.layers:
    print(i.out_shape)
        
for i in x.layers:
    if issubclass(i.__class__,nnet.synapses):
        print(np.max(np.abs(i.W.ravel())))
