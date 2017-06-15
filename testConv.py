# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:04:56 2017

@author: admin-congo
"""

import numpy as np
from sys import exit
""" The next lines will be optimized later, they are not fundamentals for the moment"""

""" END OF DEFINITION SECTION """

input1 = np.array([0,0,0,0,0,0,0,
                   0,1,2,1,1,0,0,
                   0,1,0,2,0,2,0,
                   0,2,2,1,0,2,0,
                   0,2,1,2,1,1,0,
                   0,2,2,0,1,1,0,
                   0,0,0,0,0,0,0])
    
input2 = np.array([0,0,0,0,0,0,0,
                   0,0,0,2,0,1,0,
                   0,0,2,2,2,2,0,
                   0,2,2,2,0,1,0,
                   0,0,0,0,0,1,0,
                   0,1,1,0,1,2,0,
                   0,0,0,0,0,0,0])
    
input3 = np.array([0,0,0,0,0,0,0,
                   0,2,1,2,1,2,0,
                   0,2,0,0,0,0,0,
                   0,0,0,0,1,0,0,
                   0,1,0,0,1,2,0,
                   0,1,1,0,2,1,0,
                   0,0,0,0,0,0,0])

output1 = np.array([8,5,2,
                    2,2,-2,
                    1,-3,-4])
    

output2 = np.array([5,-1,-3,
                    -2,-3,-2,
                    -2,1,-2])

weight1 = np.array([-1,-1,0,
                    -1,-1,1,
                    0,1,0,
                    -1,1,0,
                    0,1,0,
                    -1,0,1,
                    -1,-1,-1,
                    -1,1,1,
                    0,0,1])
    
bias1 = np.array([1])
weight2 = np.array([0,-1,1,
                    1,0,1,
                    1,0,0,
                    0,-1,0,
                    0,-1,0,
                    -1,-1,1,
                    1,-1,0,
                    1,0,-1,
                    0,1,1])
    
bias2 = np.array([0])

up_neurons = np.array([input1,input2,input3]).reshape(3,7,7)
weights = np.stack([weight1.reshape(3,3,3),weight2.reshape(3,3,3)])
biases = np.concatenate([bias1,bias2])
down_neurons = np.array([output1,output2]).reshape(2,3,3)
expected_output = np.array([output1,output2]).reshape(2,3,3)
stride = 2

down_neurons = conv_layer(up_neurons,down_neurons,weights,biases,stride)
print(down_neurons == expected_output)