# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:20:18 2017

@author: admin-congo

This file contains some methods to test out single layers of the NNetSim
"""
import numpy as np
import NNetSim_class2 as nnet
import mnist

"""TEST METHOD FOR FC LAYER"""
"""END OF TEST METHOD FOR FC LAYER"""

"""TEST METHOD FOR CONV LAYER"""
test_input = np.array([1,2,0,2,1,
                        2,1,0,1,0,
                        1,0,0,1,2,
                        1,0,0,2,0,
                        0,1,0,0,2,
                        2,2,1,1,2,
                        1,1,1,1,1,
                        1,1,1,0,1,
                        2,2,2,0,0,
                        2,0,1,1,2,
                        2,1,1,2,2,
                        0,0,0,1,0,
                        0,2,2,2,1,
                        1,0,2,0,2,
                        2,2,2,2,2]).reshape(1,3,5,5)

test_weights = np.array([-1, 1, 0,
                          0, 0, -1,
                          0, 0, 1,
                          0, 0, 1,
                          1, 0, 1,
                          1, 0, 1,
                          1, -1, 0,
                          -1, -1, 0,
                          1, 0, 0,
                          -1, 0, -1,
                          0, 0, 0,
                          1, 1, -1,
                          1, 1, 1,
                          0, 0, 0,
                          0, 0, 1,
                          1, -1, 0,
                          0, -1, 1,
                          -1, -1, -1]).reshape(2,3,3,3)

test_biases = np.array([1, 0])

test_output = np.array([[[ 1,  3,  0],
                          [ 7,  1, -2],
                          [ 0, -4, -6]],
                         [[ 1,  1, -2],
                          [ 5, -3,  1],
                          [ 3,  0, -6]]]).reshape(1,2,3,3)

padding = 1
stride = 2

cl = nnet.conv_layer(test_input.shape,test_weights.shape,stride,padding)

cl.W = test_weights
cl.b = test_biases

if (test_output == cl.forward(test_input)).all():
    print('Forward Convolution OK\n')
else: print('Forward Convolution not OK\n')

test_output
print(cl.backward(test_output))
"""END OF TEST METHOD FOR CONV LAYER"""

"""TEST METHOD FOR MAXPOOL LAYER"""
"""END OF TEST METHOD FOR MAXPOOL LAYER"""

"""TEST METHOD FOR ReLU LAYER"""
"""END OF TEST METHOD FOR ReLU LAYER"""


"""TEST METHOD FOR WHOLE NETWORK"""

full_set = [i for i in mnist.read_mnist('training')]
np.random.shuffle(full_set)
# Extracting labels and images, assuming that the training set is a list of tuples
labels, images = zip(*full_set)

images = np.array(images)
images = images.reshape(np.insert(images.shape,1,1))

# TRIAL 1
x = nnet.NeuralNet(0,0,10,128)
x.layers.append(nnet.conv_layer((10,1,28,28),(13,7,7),3,0))
x.layers.append(nnet.ReLU_layer(x.layers[0].out_shape))
x.layers.append(nnet.maxpool_layer(x.layers[1].out_shape,2,0))
x.layers.append(nnet.fc_layer(x.layers[2].out_shape,10))
x.num_classes = 10
x.train(images,labels,0.1)

# TRIAL 2
x = nnet.NeuralNet(0,0,10,128)
x.layers.append(nnet.fc_layer((10,784),10))
x.num_classes = 10
x.train(images,labels,0.1)

test_set = [i for i in mnist.read_mnist('testing')]
"""END OF TEST METHOD FOR WHOLE NETWORK"""