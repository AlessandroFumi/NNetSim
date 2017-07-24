# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:20:18 2017

@author: admin-congo

This file contains some methods to test out single layers of the NNetSim
"""
import numpy as np
from math import isclose
import NNetSim as nnet
import mnist

"""TEST METHOD FOR FC LAYER"""

"""END OF TEST METHOD FOR FC LAYER"""

"""TEST METHOD FOR CONV LAYER"""
#test_input = np.array([1,2,0,2,1,
#                        2,1,0,1,0,
#                        1,0,0,1,2,
#                        1,0,0,2,0,
#                        0,1,0,0,2,
#                        2,2,1,1,2,
#                        1,1,1,1,1,
#                        1,1,1,0,1,
#                        2,2,2,0,0,
#                        2,0,1,1,2,
#                        2,1,1,2,2,
#                        0,0,0,1,0,
#                        0,2,2,2,1,
#                        1,0,2,0,2,
#                        2,2,2,2,2]).reshape(1,3,5,5).astype(np.float64)
#
#test_weights = np.array([-1, 1, 0,
#                          0, 0, -1,
#                          0, 0, 1,
#                          0, 0, 1,
#                          1, 0, 1,
#                          1, 0, 1,
#                          1, -1, 0,
#                          -1, -1, 0,
#                          1, 0, 0,
#                          -1, 0, -1,
#                          0, 0, 0,
#                          1, 1, -1,
#                          1, 1, 1,
#                          0, 0, 0,
#                          0, 0, 1,
#                          1, -1, 0,
#                          0, -1, 1,
#                          -1, -1, -1]).reshape(2,3,3,3).astype(np.float64)
#
#test_biases = np.array([1, 0]).astype(np.float64)
#
#test_output = np.array([[[ 1,  3,  0],
#                          [ 7,  1, -2],
#                          [ 0, -4, -6]],
#                         [[ 1,  1, -2],
#                          [ 5, -3,  1],
#                          [ 3,  0, -6]]]).reshape(1,2,3,3).astype(np.float64)
#
#padding = 1
#stride = 2
#
#weight_shape = (2,3,3)
#cl = nnet.conv_layer(test_input.shape,weight_shape,stride,padding)
#
#cl.W = test_weights
#cl.b = test_biases
#cl.loadvalues(test_input)
#if (test_output == cl.forward()).all():
#    print('Forward Convolution OK\n')
#else: print('Forward Convolution not OK\n')
#
#""" Backward pass """
#""" If we assume that we rotate the kernel during the backpropagation step """
#X_shape = (1,1,3,3)
#W_shape = (1,2,2)
#
#cl = nnet.conv_layer(X_shape,W_shape)
#
#cl.W = np.random.randn(*cl.W_shape)
#cl.b.fill(0)
#
#test_dout = np.random.randn(*cl.out_shape)
## Producing test_dX
#v_dout = test_dout[0,0].ravel()
#v_W = cl.W[0,0].ravel()
#test_dX = np.array([
#                    v_dout[0]*v_W[3],
#                    v_dout[0]*v_W[2] + v_dout[1]*v_W[3],
#                    v_dout[1]*v_W[2],
#                    v_dout[0]*v_W[1] + v_dout[2]*v_W[3],
#                    v_dout[0]*v_W[0] + v_dout[1]*v_W[1] + v_dout[2]*v_W[2] + v_dout[3]*v_W[3],
#                    v_dout[1]*v_W[0] + v_dout[3]*v_W[2],
#                    v_dout[2]*v_W[1],
#                    v_dout[2]*v_W[0] + v_dout[3]*v_W[1],
#                    v_dout[3]*v_W[0]
#                    ]).reshape(cl.X_shape)
#del v_dout, v_W
#
##   Generate im2_col
#cl.loadvalues(cl.X)
#cl.forward()
#cl.loaderrors(test_dout)
#if (test_dX == cl.backward()).all():
#    print('Back Convolution OK\n')
#else: print('Back Convolution not OK\n')
#
#print(np.sum(np.abs(test_dX - cl.backward())))

"""END OF TEST METHOD FOR CONV LAYER"""

"""TEST METHOD FOR MAXPOOL LAYER"""
# With random numbers
#X_shape = (10,7,21,21)
#stride = 3
#padding = 0
#
#x = nnet.maxpool_layer(X_shape,stride,padding)
#
#X_in = np.random.randn(*X_shape)
#x.loadvalues(X_in)
#out = x.forward()
#x.loaderrors(out)
#X_in_back = x.backward()
#
#print( (x.reshaped_mask() == (X_in_back == X_in).astype(int)).all() )
#
## With ranged arrays
#X_shape = (1,1,6,6)
#stride = 3
#padding = 0
#X_in = np.arange(np.prod(X_shape)).reshape(X_shape)
#
#x = nnet.maxpool_layer(X_shape,stride,padding)
#x.loadvalues(X_in)
#out = x.forward()
#x.loaderrors(out)
#X_in_back = x.backward()[0,0]
#print(X_in_back)

"""END OF TEST METHOD FOR MAXPOOL LAYER"""

"""TEST METHOD FOR ReLU LAYER"""
"""END OF TEST METHOD FOR ReLU LAYER"""

"""TEST METHOD FOR SOFTMAX LAYER"""
in_shape = (1,10)
invalues = np.random.randn(*in_shape).ravel()
out = np.exp(invalues)/np.sum(np.exp(invalues))
assert(np.sum(out) > 0.9999 and np.sum(out) < 1.0001)
dout = out
jacobian = -np.outer(out,out) + out*np.eye(out.shape[0])
dX = np.dot(jacobian,dout)

print(dX)
print(np.multiply(out,dout)-np.multiply(np.sum(np.multiply(out,dout)),out))

x = nnet.softmax_layer(in_shape)

x.loadvalues(invalues)
assert ((out == x.forward()).all())
out = x.forward()
x.loaderrors(out)
dX = x.backward()
print(dX)

#Let's do it again with augmented dimensions
invalues = invalues.reshape(in_shape)
assert( (out[None,...] == np.exp(invalues)/np.sum(np.exp(invalues),axis = 1)[...,None]).all() )
out = np.exp(invalues)/np.sum(np.exp(invalues),axis = 1)[...,None]
dout = out
jacobian = - np.einsum('ij,ik->ijk',out,out) \
            + np.einsum('ij,...j->ij...',out,np.eye(out.shape[-1]))
            
print(np.einsum('ijk,ik ->ij',jacobian,dout))
"""END OF TEST METHOD FOR SOFTMAX LAYER"""

"""START OF CODE FOR GRADIENT CHECK"""
X_shape = (7,10)
out_shape = (1)

X = np.random.randn(*X_shape)
gtruth = 1

x = nnet.fc_layer(X_shape,out_shape)
x.__dict__

"""END OF CODE FOR GRADIENT CHECK"""


"""TEST METHOD FOR WHOLE NETWORK"""
#
#full_set = [i for i in mnist.read_mnist('training')]
#np.random.shuffle(full_set)
## Extracting labels and images, assuming that the training set is a list of tuples
#labels, images = zip(*full_set)
#
#images = np.array(images)
#images = images.reshape(np.insert(images.shape,1,1))
#
## TRIAL 1
#x = nnet.NeuralNet(0,0,10,128)
#x.layers.append(nnet.conv_layer((10,1,28,28),(13,7,7),3,0))
#x.layers.append(nnet.ReLU_layer(x.layers[0].out_shape))
#x.layers.append(nnet.maxpool_layer(x.layers[1].out_shape,2,0))
#x.layers.append(nnet.fc_layer(x.layers[2].out_shape,10))
#x.num_classes = 10
#x.train(images,labels,0.1)
#
## TRIAL 2
#x = nnet.NeuralNet(0,0,10,128)
#x.layers.append(nnet.fc_layer((10,784),10))
#x.num_classes = 10
#x.train(images,labels,0.1)
#
#test_set = [i for i in mnist.read_mnist('testing')]
"""END OF TEST METHOD FOR WHOLE NETWORK"""