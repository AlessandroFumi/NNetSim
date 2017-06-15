# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:32:40 2017
Initialization of the NNetSim
@author: Alessandro Fumarola
"""
import numpy as np
import NNvars

def normalize_array(input_array):
    # Returns the same array with values comprised between -1 and 1, with 
    # linear normalization
    return np.divide(input_array,np.absolute(input_array).max())

def ReLU_activation(neuron_values):
    return neuron_values.clip(min=0)

def ReLU_derivative(neuron_values):
    return (neuron_values > 0.0).astype(np.float)

def sigmoid_activation(net_input_array):
    # Returns the element-wise calculation of the sigmoid activation function
    return 1/(1 + np.exp(np.multiply(-NNvars.slope,net_input_array)))

def sigmoid_derivative(neuron_values):
    # Returns the element-wise calculation of the sigmoid derivative starting
    # from the activation value (NOT THE NET INPUT!)
    return np.multiply(neuron_values,np.subtract(1,neuron_values))

def tanh_activation(net_input_array):
    # Returns the element-wise calculation of the tanh activation function
    return 0.5*np.tanh(np.multiply(NNvars.slope,net_input_array))+0.5

def tanh_derivative(neuron_values):
    # Returns the element-wise calculation of the tanh derivative
    return np.subtract(1,np.power(neuron_values,2))

def PWL_activation(net_input_array):
    # Returns the element-wise calculation of the tanh activation function
    return np.multiply(NNvars.slope,net_input_array).clip(min=0,max=1)

def PWL_derivative(neuron_values):
    # Returns the element-wise calculation of the tanh derivative
    return np.logical_and(neuron_values > 0.0,neuron_values < 1.0).astype(np.float)

def input_layer(input_neurons, image):
    # Input layer
    # The variable input_neurons is usually denoted as neuron_values[0] while
    # image is denoted with data_in[1]
    
    # Normalized
    input_neurons = normalize_array(image.ravel())
    
    return input_neurons

def forward_inference(weights, biases, neuron_values):
    for thisLayer in range(len(neuron_values)-1):
        # We use the dot product to calculate the net input of every neuron,
        # and subsequently add the bias vector
        net_input = np.add(biases[thisLayer].ravel(),
                           np.dot(neuron_values[thisLayer],weights[thisLayer]))
        # Normalization + 'clip' routine that simulates the ReLU behaviour
        neuron_values[thisLayer+1] = activation_function(normalize_array(net_input))
    
    return neuron_values

def out_delta(output_values, output_deltas, label):
    # Pass the new delta values to the heavyweight part of the program
    # To check the correct working principle for batch size != 1
    ground_truth = np.zeros(len(output_deltas))
    ground_truth[label] = 1
    output_deltas = np.subtract(ground_truth, output_values)
    
    return output_deltas
      
def backpropagation_MSE(weights, neuron_values, delta_values):
    # Backpropagation
    for thisLayer in range(1,len(neuron_values)-1):
        # Normalized with neuron values derivatives
        net_input = np.multiply(neuron_derivative(neuron_values[-thisLayer-1]),
                                np.dot(delta_values[-thisLayer],weights[-thisLayer].transpose()))
        delta_values[-thisLayer-1] = normalize_array(net_input)
    
    return delta_values

def weight_update(weights, biases, neuron_values, delta_values, learningRate):
    # Weight update
    for thisLayer in range(len(neuron_values)-1):
        delta_weights = np.multiply(learningRate, np.outer(neuron_values[thisLayer],delta_values[thisLayer]))
        delta_biases = np.multiply(learningRate, delta_values[thisLayer]) 
        weights[thisLayer] += delta_weights
        biases[thisLayer] += delta_biases
        
    return weights, biases

def reset_neurons(neuron_values,delta_values):
    for thisLayer in range(len(neuron_values)):
        neuron_values[thisLayer].fill(0)
        delta_values[thisLayer-1].fill(0)
        
    return neuron_values, delta_values

def print_training_info(im_counter,im_correct):
    # Display training accuracy
    trainAccuracy = np.multiply(100,np.divide(im_correct,im_counter))
    info_msg = 'Accuracy: %f%%' % trainAccuracy
    print(info_msg)
    return trainAccuracy

neuron_type_dictionary = {
        'ReLU' : (ReLU_activation, ReLU_derivative),
        'sigmoid' : (sigmoid_activation, sigmoid_derivative),
        'tanh' : (tanh_activation, tanh_derivative),
        'PWL' : (PWL_activation, PWL_derivative)
        }

(activation_function, neuron_derivative) = neuron_type_dictionary[NNvars.neuron_type]
# Overrides the previous definition of the backpropagation routine if the 
# cross-entropy cost function is used instead of the MSE. 
cost_function_dictionary = {
        'MSE' : backpropagation_MSE
        }
backpropagation = cost_function_dictionary[NNvars.cost_function]