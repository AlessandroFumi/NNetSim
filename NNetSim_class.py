# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:48:20 2017
This file contains an experimental version of NNSim using Python Classes

Reutilization of previous code shall be maximized

@author: Alessandro Fumarola
"""
import numpy as np
"""NECESSARY FUNCTIONS"""
def gen_indexes(input_size,kernel_size,stride):
    # Total number of slidings
    # We could devise this function to be called just at the beginning
    num_slidings = int((input_size-kernel_size) / stride + 1)
    out_indexes = np.array([])
    for i in np.arange(num_slidings):
        out_indexes = np.append(out_indexes,np.arange(i*stride ,i*stride+kernel_size))
    
    sorting_indexes = out_indexes.argsort().argsort()
    (ignore,reps) =  np.unique(out_indexes, return_counts = True)

    return num_slidings, reps, sorting_indexes

def fast3dconv(up_neurons,out_size,weights,stride):
    # Extracting sizes
    input_size = up_neurons.shape[-1]
    kernel_size = weights.shape[-1]
    # Padding the upstream neurons
    pad_width = int(( (out_size-1) * stride + kernel_size - input_size) / 2)
    npad = ((0,0),(pad_width,pad_width),(pad_width,pad_width))
    # Extracting sizes
    input_size += 2*pad_width
    # We have two different versions here
    num_slidings, reps, sorting_indexes = gen_indexes(input_size,kernel_size,stride)
    # Padding
    up_neurons = np.pad(up_neurons, npad, 'constant', constant_values = 0)
    # Replicate and ordering rows and columns
    up_neurons = np.repeat(up_neurons,reps,axis=-1)
    up_neurons = up_neurons[...,sorting_indexes]
    up_neurons= np.repeat(up_neurons,reps,axis=-2)
    up_neurons = up_neurons[...,sorting_indexes,:]
    # Splitting in equal parts
    up_neurons= np.array(np.split(up_neurons,num_slidings,axis=-1))
    up_neurons= np.array(np.split(up_neurons,num_slidings,axis=-2))

    #Replicating weights
    weights = np.tile(weights,(num_slidings,num_slidings))
    weights = np.array(np.split(weights,num_slidings,axis=-1))
    weights = np.array(np.split(weights,num_slidings,axis=-2))
    #Here we have the assumption that weights are a 4D tensor turned 6D
    weights = weights.transpose([2,0,1,3,4,5])
    
    # Einstein product: convolution without biasing
    down_neurons = np.einsum('...ijk,...ijk->...', up_neurons , weights)
    
    return down_neurons

def fiveDtothreeD(tensor):
    tensor = tensor.transpose([1,3,2,4,0])
    tensor = np.concatenate(tensor, axis = 0)
    tensor = tensor.transpose([1,2,3,0])
    tensor = np.concatenate(tensor, axis = 0)
    tensor = tensor.transpose([1,2,0])
    return tensor

""" FORWARD EVALUATION """
def fc_layer(up_neurons, weights, biases):
    # We use the dot product to calculate the net input of every neuron,
    # and subsequently add the bias vector
    down_neurons = np.add(biases.ravel(),
                           np.dot(up_neurons,weights))
    return down_neurons

def ReLU_layer(up_neurons):
    # Normalization + 'clip' routine that simulates the ReLU behaviour
    down_neurons = up_neurons.clip(min=0)
    return down_neurons

def conv_layer(up_neurons,out_size,weights,biases,stride):
    # Do the convolution and add the biases
    down_neurons = fast3dconv(up_neurons,out_size,weights,stride) + biases[...,None,None]
    return down_neurons

def maxpool_layer(up_neurons,down_neurons):
    # Extracting sizes
    out_size = down_neurons.shape[-1]
    # Divide the input volume slices by NxN using the filter dimension
    up_neurons = np.array(np.split(up_neurons,out_size,axis=-1))
    up_neurons = np.array(np.split(up_neurons,out_size,axis=-2))
    # Pool the maximum from correct axes and reorder
    down_neurons = up_neurons.max(axis=(3,4)).transpose([2,0,1])
    # Build boolean mask for error propagation
    up_neurons = up_neurons.transpose([2,0,1,3,4])
    bool_mask = (up_neurons == down_neurons[...,None,None]).astype(np.int)
    # If we want to reproduce the size of input layer, to be checked
#    bool_mask = fiveDtothreeD(bool_mask)
    return down_neurons,bool_mask

"""LAYER FUNCTIONALITY, BACK"""
def out_delta(output_values, output_deltas, label):
    # Pass the new delta values to the heavyweight part of the program
    # To check the correct working principle for batch size != 1
    output_deltas.fill(0)
    ground_truth = np.zeros(len(output_deltas))
    ground_truth[label] = max(output_values)
    output_deltas += np.subtract(ground_truth, output_values)
    
    return output_deltas

def fc_layer_back(down_errors, weights):
    up_errors = np.dot(down_errors, weights.transpose())
    return up_errors

def ReLU_layer_back(down_errors,up_neurons):
    # Normalization + 'clip' routine that simulates the ReLU behaviour
    up_errors = np.multiply(down_errors,(up_neurons > 0.0).astype(np.float))
    return up_errors

def conv_layer_back(down_errors,up_size,weights,stride):
    weights = np.flip(weights,axis=-1)
    weights = np.flip(weights,axis=-2)
    # The backpropagation of a convolutional layer is still a convolution
    up_errors = fast3dconv(down_errors,up_size,weights,stride)
    return up_errors

def maxpool_layer_back(up_errors,down_errors,bool_mask):
    up_errors = np.multiply(down_errors[...,None,None], bool_mask)
    up_errors = fiveDtothreeD(up_errors)
    return up_errors

'''TEST METHOD FOR CONV LAYER, NOT USED SO FAR '''
def test_conv(self):
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
    
    self.__init__(test_input.shape,test_weights.shape,stride,padding)
    
    self.W = test_weights
    self.b = test_biases
    self.W_shape = self.W.shape
    
    if (test_output == self.forward(test_input)).all():
        print('Forward Convolution OK\n')
    else: print('Forward Convolution not OK\n')
    
    print(self.backward(test_output))
    
    return
