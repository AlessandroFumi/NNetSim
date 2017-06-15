# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 18:53:52 2017

@author: admin-congo
"""

# Testing
print('TESTING BEGINS:')
im_counter = 0
im_correct = 0
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

    # Printing results
    if np.mod(im_counter,NNvars.numCheckIterations) == 0:
        print('Accuracy = %f \t %%' % np.multiply(100,np.divide(im_correct,im_counter)))
print('TESTING ENDS, final accuracy = %f \t %% \n' % (np.multiply(100,np.divide(im_correct,im_counter))))