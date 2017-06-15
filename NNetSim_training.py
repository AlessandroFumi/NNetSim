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
import matplotlib.pyplot as plt
# Let's try to save the accuracies and subsequently plot them
list_trainacc = []
# Training
print('TRAINING BEGINS:')
for epoch_count in range(NNvars.numEpochs):
    print('Epoch %d:' % (epoch_count+1))
    # The following are all temporary variables used to display relevant
    # information about training
    im_counter = 0
    im_correct = 0
    trainAccuracy = 0
    verificationAccuracy = 0
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
        
        # Compute expected neuron values
        NNvars.expected_values = [NNvars.neuron_values[i+1] + NNvars.delta_values[i]
                                    for i in range(len(NNvars.delta_values))]

        # Checking correctness    
        if data_in[0] != np.argmax(NNvars.neuron_values[-1]):
            # Weight update
            (NNvars.weights, NNvars.biases) = NNfuncs.weight_update(NNvars.weights,NNvars.biases,NNvars.neuron_values,NNvars.delta_values, NNvars.learningRate)
        # Reset neuron and delta values
        (NNvars.neuron_values, NNvars.delta_values) = NNfuncs.reset_neurons(NNvars.neuron_values,NNvars.delta_values)            
        
        # Print useful information
        if np.mod(im_counter,NNvars.numCheckIterations) == 0:
            trainAccuracy = NNfuncs.print_training_info(im_counter,im_correct)
            list_trainacc.append(trainAccuracy)

    print('Epoch %d, accuracy on training set = %f %%' % (epoch_count+1,trainAccuracy))
#    for i in range(len(NNvars.weights)):
#        plt.show(plt.pcolor(NNvars.weights[i]))
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
    
    print('Accuracy on verification set = %f %%' 
          % (np.multiply(100,np.divide(im_correct,im_counter))))

    # Reset everything for test
    im_counter = 0
    im_correct = 0
    
    # Test set
    for data_in in data_prep.test_set:
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
    
    print('Accuracy on test set = %f %% \n' 
          % (np.multiply(100,np.divide(im_correct,im_counter))))
    
print('TRAINING ENDS')