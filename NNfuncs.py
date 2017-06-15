# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:32:40 2017
Initialization of the NNetSim
@author: Alessandro Fumarola
"""
#from sys import exit
#from scipy import misc 
#from skimage import io

import numpy as np

""" MISC FUNCTIONS NEEDED """
#def checking():
#    # All this checking is done because this function is really buggy
#    if np.mod(input_size-kernel_size,stride) or kernel_size > input_size or stride > input_size:
#        print("The input, kernel and stride size have to satisfy certain relations.")
#        exit(1)
#    elif stride > kernel_size:
#        print("We're not yet ready for dilated convolutions.")
#        exit(1)
#    return isgood

def threeDtofiveDconv(tensor, reps, sorting_indexes):
    num_slidings = reps.sum()
    # Replicate and ordering rows and columns
    tensor = np.repeat(tensor,reps,axis=-1)
    tensor = tensor[...,sorting_indexes]
    tensor= np.repeat(tensor,reps,axis=-2)
    tensor = tensor[...,sorting_indexes,:]
    # Splitting in equal parts
    tensor= np.array(np.split(tensor,num_slidings,axis=-1))
    tensor= np.array(np.split(tensor,num_slidings,axis=-2))
    return tensor

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
"""END OF IMPLEMENTATION WITH 5D TENSORS"""

"""IMPLEMENTATION OF CONVOLUTION WITH IM2COL AND FANCY INDEXING"""
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_width) % stride == 0
  out_height = int( (H + 2 * padding - field_height) / stride + 1 )
  out_width = int( (W + 2 * padding - field_width) / stride + 1 )

  i0 = np.repeat(np.arange(field_height), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (p, p), (p, p)), mode='constant')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

  cols = x_padded[k, i, j]
  C = x.shape[0]
  cols = cols.reshape(field_height * field_width * C, -1)
  return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
  """ An implementation of col2im based on fancy indexing and np.add.at """
  C, H, W = x_shape
  H_padded, W_padded = H + 2 * padding, W + 2 * padding
  x_padded = np.zeros((C, H_padded, W_padded), dtype=cols.dtype)
  k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
  cols_reshaped = cols.reshape(C * field_height * field_width, -1)
  np.add.at(x_padded, (k, i, j), cols_reshaped)
  if padding == 0:
    return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]

def im2col_conv(x_in,x_out,filters,biases,stride=1,padding=1):
    field_height, field_width = filters.shape[-2:]
    # Convert x to matrix
    in_cols = im2col_indices(x_in, field_height, field_width, padding, stride)
    filters = filters.reshape(filters.shape[0],-1)
    out_cols = np.dot(filters,in_cols)+biases[...,None]
    return col2im_indices(out_cols, x_out.shape,
                          field_height, field_width, padding, stride)

"""END OF IMPLEMENTATION OF CONVOLUTION WITH IM2COL"""

"""NEURON ACTIVATION FUNCTIONS AND DERIVATIVE!"""
def ReLU_activation(net_input_array):
    return net_input_array.clip(min=0)

def ReLU_derivative(neuron_values):
    return (neuron_values > 0.0).astype(np.float)

"""LAYER FUNCTIONALITY, FORWARD"""
def fc_layer(up_neurons, weights, biases):
    # We use the dot product to calculate the net input of every neuron,
    # and subsequently add the bias vector
    down_neurons = np.add(biases.ravel(),
                           np.dot(up_neurons,weights))
    return down_neurons

def ReLU_layer(up_neurons):
    # Normalization + 'clip' routine that simulates the ReLU behaviour
    down_neurons = ReLU_activation(up_neurons)
    return down_neurons

def conv_layer(up_neurons,down_neurons,weights,biases,stride):
    out_size = down_neurons.shape[-1]
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
    up_errors = np.multiply(down_errors,ReLU_derivative(up_neurons))
    return up_errors

def conv_layer_back(up_errors,down_errors,weights):
    weights = np.flip(weights,axis=-1)
    weights = np.flip(weights,axis=-2)
    # The backpropagation of a convolutional layer is still a convolution
    up_errors = fast3dconv(down_errors,up_errors, weights)
    return up_errors

def maxpool_layer_back(up_errors,down_errors,bool_mask):
    up_errors = np.multiply(down_errors[...,None,None], bool_mask)
    up_errors = fiveDtothreeD(up_errors)
    return up_errors

"""WEIGHT UPDATES"""
def fc_layer_wu(up_neurons,down_errors,learningRate):
    weight_change = np.multiply(learningRate, np.outer(up_neurons,down_errors))
    bias_change = np.multiply(learningRate, down_errors) 
    return weight_change, bias_change

def conv_layer_wu(up_neurons, down_errors, learningRate):
    
    weight_change = np.array(0)
    bias_change = np.array(0)
    return weight_change, bias_change
"""DATA FLOW CONTROLLERS"""
def input_layer(input_neurons, image):
    # Input layer
    # The variable input_neurons is usually denoted as neuron_values[0] while
    # image is denoted with data_in[1]
    
    # Normalized
    input_neurons = image.ravel()
    
    return input_neurons

def forward_inference(weights, biases, neuron_values):
    # Here we'll put the control for the forward evaluation
    
    return neuron_values

      
def backpropagation(weights, neuron_values, delta_values):
    # Controller for backpropagation
    return delta_values

def reset_neurons(neuron_values,delta_values):
    # This is used to reset neuron value
    for thisLayer in range(len(neuron_values)):
        neuron_values[thisLayer].fill(0)
        delta_values[thisLayer-1].fill(0)
        
    return neuron_values, delta_values

"""CALCULATE AND DISPLAY USEFUL INFORMATION"""
def total_error_MSE(delta_values,error_values):
    #Returns the total error for every layer in the neural network
    for i in range(len(delta_values)):
        error_values[i] += 0.5*np.sum(np.power(delta_values[i],2)) 
    return error_values

def print_training_info():
    return 0