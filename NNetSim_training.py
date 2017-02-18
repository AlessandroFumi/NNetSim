# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:03:45 2017
This file shall act just as a controller for the information flow:
    Counting the images
    Counting the iterations
    Checking accuracy
    etc...
@author: admin-congo
"""
import numpy as np
import NNvars, NNfuncs, data_prep
import shelve, os

# Training
print('TRAINING BEGINS:')
for epoch_count in range(NNvars.numEpochs):
    print('Epoch %d:' % (epoch_count+1))
    # The following are all temporary variables used to display relevant
    # information about training
    im_counter = 0
    im_correct = 0
    trainAccuracy = 0
    previoustrainAccuracy = 0
    verificationAccuracy = 0
    previousverificationAccuracy = 0
    error_values = [0] * len(NNvars.delta_values)
    sumWeightsChange = [0] * len(NNvars.weights)
    
    # Cycling over the mnist dataset
    for data_in in data_prep.training_set:
        # Number of processed images
        im_counter += 1
        
        # Input layer
        NNvars.neuron_values[0] = NNfuncs.input_layer(NNvars.neuron_values[0], data_in[1])
        
        # Forward inference
        NNvars.neuron_values = NNfuncs.forward_inference(NNvars.weights,NNvars.biases,NNvars.neuron_values)

        # Checking correctness    
        if data_in[0] == np.argmax(NNvars.neuron_values[-1]):
            im_correct += 1
            
        # Generation of output layer's delta
        NNvars.delta_values[-1] = NNfuncs.out_delta(NNvars.neuron_values[-1],NNvars.delta_values[-1],data_in[0])
        
        # Backpropagation
        NNvars.delta_values = NNfuncs.backpropagation(NNvars.weights, NNvars.neuron_values, NNvars.delta_values)
        
        # Compute error values (MSE or xentropy)        
        error_values = NNfuncs.total_error(NNvars.delta_values,error_values)
            
        if (np.mod(im_counter,NNvars.batchSize) == 0):
            # Weight update
            (NNvars.weights, NNvars.biases, weightsChange, biasesChange) = NNfuncs.weight_update(NNvars.weights,NNvars.biases,NNvars.neuron_values,NNvars.delta_values, NNvars.learningRate)
            # Accumulate the average weight change per layer (summed all over the examples)
            sumWeightsChange = [sumWeightsChange[i] + weightsChange[i] + biasesChange[i] for i in range(len(sumWeightsChange))]
            # Reset neuron and delta values
            (NNvars.neuron_values, NNvars.delta_values) = NNfuncs.reset_neurons(NNvars.neuron_values,NNvars.delta_values)            
        
        
        # Display relevant information
        if np.mod(im_counter,NNvars.numCheckIterations) == 0:
            # Display training accuracy
            previoustrainAccuracy = trainAccuracy
            trainAccuracy = np.multiply(100,np.divide(im_correct,im_counter))
            info_msg = 'Accuracy: %f \t Errors:\t' % trainAccuracy
            
            # Display cost function for every delta layer
            for i in range(len(error_values)):
                info_msg += ('%f \t') % (np.divide(error_values[i],im_counter))
            
#            info_msg += '\nAverage weight change per neuron per example:'
            # Display average weight update for every layer
#            for i in range(len(sumWeightsChange)):
#                info_msg += ('%f \t') % (np.divide(sumWeightsChange[i].mean(),im_counter))
            print(info_msg)
                    
    print('Epoch %d, accuracy on training set = %f %%' % (epoch_count+1,trainAccuracy))
    
    # Learning rate decay
    NNvars.learningRate *= NNvars.learningRateDecay
    #Let's shuffle the training set
    np.random.shuffle(data_prep.training_set)

    # Reset everything for verification
    im_counter = 0
    im_correct = 0
    
    # Verification set
    for data_in in data_prep.verification_set:
        # Number of processed images
        im_counter += 1
        
        # Input layer
        NNvars.neuron_values[0] = NNfuncs.input_layer(NNvars.neuron_values[0], data_in[1])
        
        # Forward inference
        NNvars.neuron_values = NNfuncs.forward_inference(NNvars.weights,NNvars.biases,NNvars.neuron_values)

        # Checking correctness    
        if data_in[0] == np.argmax(NNvars.neuron_values[-1]):
            im_correct += 1
            
        # Reset neuron and delta values
        (NNvars.neuron_values, NNvars.delta_values) = NNfuncs.reset_neurons(NNvars.neuron_values,NNvars.delta_values)
    
    
    # Display training accuracy
    previoustrainAccuracy = verificationAccuracy
    verificationAccuracy = np.multiply(100,np.divide(im_correct,im_counter))
    
    print('Accuracy on verification set = %f %% \n' 
          % (np.multiply(100,np.divide(im_correct,im_counter))))
    
    
print('TRAINING ENDS')