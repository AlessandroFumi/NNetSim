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
# Extracting labels and images, assuming that the training set is a list of tuples
labels, images = zip(*full_set)

images = np.array(images)
images = images.reshape(np.insert(images.shape,1,1))


test_set = [i for i in mnist.read_mnist('testing')]
# TRIAL 1
x = NeuralNet(0,0,10,128)
x.layers.append(conv_layer((10,1,28,28),(13,7,7),3,0))
x.layers.append(ReLU_layer(x.layers[0].out_shape))
x.layers.append(maxpool_layer(x.layers[1].out_shape,2,0))
x.layers.append(fc_layer(x.layers[2].out_shape,10))
x.num_classes = 10
x.train(images,labels,0.1)

# TRIAL 2
x = NeuralNet(0,0,10,128)
x.layers.append(fc_layer((10,784),10))
x.num_classes = 10
x.train(images,labels,0.1)

layerList =[conv_layer,ReLU_layer,maxpool_layer,fc_layer]
paramList = (((10,1,28,28),(13,7,7),3,0),(),(2,0),(10))
