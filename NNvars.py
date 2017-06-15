# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:12:55 2017
Initialization script for neural NNetSim
Here we correctly inizialize all the hyperparameters, neurons, synapses etc.
@author: Alessandro Fumarola
"""
from sys import exit
#from scipy import misc
#from skimage import io
import numpy as np

# System parameters for neural network, to be decided by the experimenter every
# time
# Used in the weight update function
learningRate = 1
# Every epoch, the learningrate will be decreased
learningRateDecay = 0.95
# Used in every activation function except for the ReLU
# Sets the steepness of the activation function
slope = 10
# Max weight value
maxWeight = 10
# Number of allowed synaptic states
numStates = 100
# Hyperparameters for training
# How many images from MNIST?
numSamples = 5000
numVerification = 10000
# How many training steps per image
numEpochs = 100
# How many images will be processed at once?
# As of 14.02.2017 this feature does not work properly
batchSize = 1
# Two possible initialization for weights and biases, 'gaussian' or 'uniform'.
# Always between -1 and 1
initialization = 'uniform'

# Topology of the network, put the number of neurons separated by a comma, as example
# numNeurons = np.array([784,250,150,10])
numNeurons = np.array([784,10])
# Type of neurons, soon we'll support different types for different layers
neuron_type = 'sigmoid'
# Type of cost function
cost_function = 'MSE'

"""THE FOLLOWING CODE CAN REMAIN UNCHANGED"""
# Information about training will be printed ~10 times for every Epoch
numCheckIterations = np.round(numSamples/10)

#One bias neuron per layer should be enough
biasNeurons = np.ones(len(numNeurons)-1).astype(np.int)

# Let's create the weight matrix and the bias vector for every layer
numSynapses = np.array([numNeurons[i]*numNeurons[i+1] for i in range(len(numNeurons)-1)])
numBiases = np.array([biasNeurons[i]*numNeurons[i+1] for i in range(len(numNeurons)-1)])

# Let's initialize the neuron values and delta_values to zero
neuron_values = [np.zeros(numNeurons[i]) for i in range(len(numNeurons))]
expected_values = [np.zeros(numNeurons[i]) for i in range(1,len(numNeurons))]
delta_values = [np.zeros(numNeurons[i]) for i in range(1,len(numNeurons))]

initialization_dictionary = {
        'uniform' : np.random.uniform,
        'gaussian' : np.random.randn,
        'last' : None
        }


if initialization not in initialization_dictionary:
    info_msg = 'Synapse and bias initialization has to be one of the following:\n %s' % ' ,'.join(['%s'%i for i in initialization_dictionary])
    print(info_msg)
    exit(1)    

# Weights and biases are initialized to random values (according to the distribution)
# specified by the initialization variable
if 'uniform' in initialization:
    weights = [np.random.uniform(-1,1,numSynapses[i]).reshape(numNeurons[i],numNeurons[i+1]) for i in range(len(numNeurons)-1)]
    biases = [np.random.uniform(-1,1,numBiases[i]).reshape(biasNeurons[i],numNeurons[i+1]) for i in range(len(numNeurons)-1)]
elif 'gaussian' in initialization:
    weights = [np.random.randn(numNeurons[i],numNeurons[i+1]) for i in range(len(numNeurons)-1)]
    biases = [np.random.randn(biasNeurons[i],numNeurons[i+1]) for i in range(len(numNeurons)-1)]

print('SETUP for neural network:')
print('learning rate = %f, learning rate decay = %f %%, neuron type = %s, number of input images = %d, epochs = %d' %
      (learningRate,100*learningRateDecay, neuron_type, numSamples, numEpochs))