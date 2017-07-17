# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:55:25 2017

@author: admin-congo
"""
import numpy as np
import NNetSim as nnet
import mnist
np.set_printoptions(threshold=np.nan)


full_set = [i for i in mnist.read_mnist('training')]
np.random.shuffle(full_set)
full_set = full_set[:5000]
# Extracting labels and images, assuming that the training set is a list of tuples

test_set = [i for i in mnist.read_mnist('testing')]

# Let's try to see what shape should the NeuralNet input dictionary have
# We need to add commas at the end of single numbers to make them tuples
# TRIAL 1
NeuralNetParams = {
        'layerList': [nnet.fc_layer,nnet.fc_layer,nnet.fc_layer],
        'layerParamList' : [( (5,784),(250,) ),((125,)),((10,))]
        }
# TRIAL 2
NeuralNetParams = {
        'layerList': [nnet.softmax_layer,nnet.fc_layer],
        'layerParamList' : [((5,784),), (10,)]
        }
# TRIAL 3
NeuralNetParams = {
        'layerList': [nnet.fc_layer,nnet.softmax_layer],
        'layerParamList' : [((5,784) , (10,)),()]
        }
# TRIAL 4
NeuralNetParams = {
        'layerList': [nnet.norm_layer, nnet.fc_layer, nnet.softmax_layer],
        'layerParamList' : [((5,784),),( (10), ), ( ) ]
        }

# TRIAL 1
#x = NeuralNet(0,0,10,128)
#x.layers.append(conv_layer((10,1,28,28),(13,7,7),3,0))
#x.layers.append(ReLU_layer(x.layers[0].out_shape))
#x.layers.append(maxpool_layer(x.layers[1].out_shape,2,0))
#x.layers.append(fc_layer(x.layers[2].out_shape,10))
#x.num_classes = 10
#x.train(images,labels,0.1)
#
## TRIAL 2
#x = NeuralNet(0,0,10,128)
#x.layers.append(fc_layer((10,784),10))
#x.num_classes = 10
#x.train(images,labels,0.1)
#
#layerList =[conv_layer,ReLU_layer,maxpool_layer,fc_layer]
#paramList = (((10,1,28,28),(13,7,7),3,0),(),(2,0),(10))

x = nnet.NeuralNet()
x.add_layer(nnet.norm_layer, (100,784))
x.add_layer(nnet.fc_layer, (10,))
x.add_layer(nnet.sigmoid_layer, () )

#x.train(full_set)

for i in x.layers:
    if issubclass(i.__class__,nnet.synapses):
        print(np.average(i.W.ravel()))
