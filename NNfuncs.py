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
from scipy import ndimage, misc

import NNvars
"""NEURON ACTIVATION FUNCTIONS AND DERIVATIVE!"""
def net_input(weights, biases, upstream_values,normalization_values):
    # Returns the same array with values comprised between -1 and 1, with 
    # linear normalization
    return np.divide(np.add(biases.ravel(),np.dot(upstream_values,weights)),
              normalization_values.sum()+1)

def ReLU_activation(neuron_values):
    return neuron_values.clip(min=0)

def ReLU_derivative(neuron_values):
    return (neuron_values > 0.0).astype(np.float)

def sigmoid_activation(net_input_array):
    # Returns the element-wise calculation of the sigmoid activation function
    return 1/(1 + np.exp(np.multiply(-NNvars.slope_sigmoid,net_input_array)))

def sigmoid_derivative(neuron_values):
    # Returns the element-wise calculation of the sigmoid derivative starting
    # from the activation value (NOT THE NET INPUT!)
    return np.multiply(neuron_values,np.subtract(1,neuron_values))

def tanh_activation(net_input_array):
    # Returns the element-wise calculation of the tanh activation function
    return 0.5*np.tanh(np.multiply(NNvars.slope_tanh,net_input_array))+0.5

def tanh_derivative(neuron_values):
    # Returns the element-wise calculation of the tanh derivative
    return np.subtract(1,np.power(neuron_values,2))

def PWL_activation(net_input_array):
    # Returns the element-wise calculation of the tanh activation function
    return np.multiply(NNvars.slope_PWL,net_input_array).clip(min=0,max=1)

def PWL_derivative(neuron_values):
    # Returns the element-wise calculation of the tanh derivative
    return np.logical_and(neuron_values > 0.0,neuron_values < 1.0).astype(np.float)

"""FULLY CONNECTED NEURAL NETWORKS FUNCTIONALITY"""
def input_layer(input_neurons, image):
    # Input layer
    # The variable input_neurons is usually denoted as neuron_values[0] while
    # image is denoted with data_in[1]
    
    # Normalized
    input_neurons = np.divide(image.ravel(), image.ravel().max())
    
    return input_neurons

def forward_inference(weights, biases, neuron_values):
    for thisLayer in range(len(neuron_values)-1):
        # We use the dot product to calculate the net input of every neuron,
        # and subsequently add the bias vector
        norm_input = net_input(weights[thisLayer], biases[thisLayer],
                               neuron_values[thisLayer],neuron_values[thisLayer])
        # Normalization + 'clip' routine that simulates the ReLU behaviour
        neuron_values[thisLayer+1] += activation_function(norm_input)
    
    return neuron_values

def out_delta(output_values, output_deltas, label):
    # Pass the new delta values to the heavyweight part of the program
    # To check the correct working principle for batch size != 1
    output_deltas.fill(0)
    ground_truth = np.zeros(len(output_deltas))
    ground_truth[label] = 1
    output_deltas += np.multiply(neuron_derivative(output_values),np.subtract(ground_truth, output_values))
    
    return output_deltas
      
def backpropagation_MSE(weights, neuron_values, delta_values):
    # Backpropagation
    for thisLayer in range(1,len(neuron_values)-1):
        # Normalized with neuron values derivatives
        norm_input = np.multiply(neuron_derivative(neuron_values[-thisLayer-1]),
                                 net_input(weights[-thisLayer].transpose(),np.zeros(1),
                                           delta_values[-thisLayer],np.zeros(1))
                                 )
        delta_values[-thisLayer-1] += norm_input
    
    return delta_values

def backpropagation_xentropy(weights, neuron_values, delta_values):
    # Backpropagation
    for thisLayer in range(1,len(neuron_values)-1):
        # Normalized with neuron values derivatives
        norm_input = net_input(weights[-thisLayer].transpose(),np.zeros(1),
                               delta_values[-thisLayer],neuron_values[-thisLayer])
        delta_values[-thisLayer-1] += norm_input
    
    return delta_values

def weight_update(weights, biases, neuron_values, delta_values, learningRate):
    delta_weights = [0]*(len(neuron_values)-1)
    delta_biases = [0]*(len(neuron_values)-1)
    # Weight update
    for thisLayer in range(len(neuron_values)-1):
        delta_weights[thisLayer] = np.multiply(learningRate, np.outer(neuron_values[thisLayer],delta_values[thisLayer]))
        delta_biases[thisLayer] = np.multiply(learningRate, delta_values[thisLayer]) 
        weights[thisLayer] += delta_weights[thisLayer]
        biases[thisLayer] += delta_biases[thisLayer]
        #weights[thisLayer] = weights[thisLayer].clip(min=-1,max=1)
        #biases[thisLayer] = biases[thisLayer].clip(min=-1,max=1)
        
    return weights, biases, delta_weights, delta_biases

def reset_neurons(neuron_values,delta_values):
    for thisLayer in range(len(neuron_values)):
        neuron_values[thisLayer].fill(0)
        delta_values[thisLayer-1].fill(0)
        
    return neuron_values, delta_values
"""CONVOLUTIONAL NEURAL NETWORKS FUNCTIONALITY"""
def fc_layer(up_neurons, down_neurons, weights, biases):
    # We use the dot product to calculate the net input of every neuron,
    # and subsequently add the bias vector
    norm_input = net_input(weights, biases,up_neurons,up_neurons)
    # The apply the activation functions to the net input
    down_neurons += activation_function(norm_input)

    return down_neurons

def conv_layer(up_neurons, down_neurons, weights, biases, pad_width):
    # In conv layers neurons are three dimensional tensors, exactly as weights
    # and down neurons. We could put some constraints on the sizes but too expensive
    ndimage.convolve

"""CALCULATE AND DISPLAY USEFUL INFORMATION"""
def total_error_xentropy(delta_values,error_values):
    #Returns the total error for every layer in the neural network
    for i in range(len(delta_values)):
        error_values[i] += np.sum(np.power(delta_values[i],2)) 
    return error_values

def total_error_MSE(delta_values,error_values):
    #Returns the total error for every layer in the neural network
    for i in range(len(delta_values)):
        error_values[i] += 0.5*np.sum(np.power(delta_values[i],2)) 
    return error_values

def print_training_info():
    return 0

"""DICTIONARIES"""
neuron_type_dictionary = {
        'ReLU' : (ReLU_activation, ReLU_derivative),
        'sigmoid' : (sigmoid_activation, sigmoid_derivative),
        'tanh' : (tanh_activation, tanh_derivative),
        'PWL' : (PWL_activation, PWL_derivative)
        }

(activation_function, neuron_derivative) = neuron_type_dictionary[NNvars.neuron_type]
# LayerType dictionary
layerType_dictionary = {
        'CONV' : (),
        'ReLU' : (),
        'POOL' : (),
        'FC' : ()
        }
# Overrides the previous definition of the backpropagation routine if the 
# cross-entropy cost function is used instead of the MSE. 
cost_function_dictionary = {
        'MSE' : (backpropagation_MSE,total_error_MSE),
        'xentropy' : (backpropagation_xentropy,total_error_xentropy)
        }
(backpropagation, total_error) = cost_function_dictionary[NNvars.cost_function]