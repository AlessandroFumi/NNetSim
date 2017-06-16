import numpy as np
import sys

"""FUNCTIONS FOR EFFICIENT 2D CONVOLUTION"""

def gen_indexes(input_size,kernel_size,stride):
    """Remember, input_shape is a tuple with the complete description of the input
    tensor while input_size refers just to the width/height of the image you're trying to process"""
    # Total number of slidings
    # We could devise this function to be called just at the beginning
    num_slidings = int((input_size-kernel_size) / stride + 1)
    out_indexes = np.array([])
    for i in np.arange(num_slidings):
        out_indexes = np.append(out_indexes,np.arange(i*stride ,i*stride+kernel_size))
    
    sorting_indexes = out_indexes.argsort().argsort()
    (ignore,reps) =  np.unique(out_indexes, return_counts = True)

    return num_slidings, reps, sorting_indexes

def fast3dconv(in_values,weights,sorting_indexes,reps,num_slidings):
    # Replicate and ordering rows and columns
    in_values = np.repeat(in_values,reps,axis=-1)
    in_values = in_values[...,sorting_indexes]
    in_values= np.repeat(in_values,reps,axis=-2)
    in_values = in_values[...,sorting_indexes,:]
    # Splitting in equal parts
    in_values= np.array(np.split(in_values,num_slidings,axis=-1))
    in_values= np.array(np.split(in_values,num_slidings,axis=-2))

    #Replicating weights
    weights = np.tile(weights,(num_slidings,num_slidings))
    weights = np.array(np.split(weights,num_slidings,axis=-1))
    weights = np.array(np.split(weights,num_slidings,axis=-2))
    #Here we have the assumption that weights are a 4D tensor turned 6D
    weights = weights.transpose([2,0,1,3,4,5])
    
    # Einstein product: convolution without biasing
    out_values = np.einsum('...ijk,...ijk->...', in_values , weights)
    
    return out_values
"""END OF FUNCTIONS FOR EFFICIENT 2D CONVOLUTION"""

"""LAYER CLASSES DEFINITION"""

class layer(object):
    
    def __init__(self,in_shape,out_shape,weight_shape,bias_shape):
        #   Common properties of every layers
#        self.in_data = np.zeros(in_shape)
#        self.in_err = np.zeros(in_shape)
#        self.out_data = np.zeros(out_shape)
#        self.out_err = np.zeros(out_shape)
#        self.weights = np.random.randn(np.prod(weight_shape)).reshape(weight_shape)
#        self.biases = np.random.randn(bias_shape)

        # Maybe we should just save the weights, biases and dimensions
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.weight_shape = weight_shape
        self.bias_shape = bias_shape
        self.weights = np.random.randn(np.prod(weight_shape)).reshape(weight_shape)
        self.biases = np.random.randn(bias_shape)

class fc_layer(layer):
    def __init__(self,in_shape,out_shape):
        # Input shape = (num. input neurons)
        # Output shape = (num. output neurons)
        layer.__init__(self,in_shape,out_shape,(in_shape,out_shape),out_shape)
        del self.npad

    def forward(self,in_values):
        # We use the dot product to calculate the net input of every neuron,
        # and subsequently add the bias vector
        out_values = np.add(self.biases.ravel(),
                           np.dot(in_values.ravel(),self.weights))
        return out_values
        
class conv_layer(layer):
    def __init__(self,in_shape,filter_shape,stride=1,padding=0):
        # Input shape = (num. layers, input height, input width)
        # Filter shape = (num. filters, filter height, filter width)
        
        outd = (in_shape[-1]+padding-filter_shape[-1])/stride+1
        # Checking dimensional consistency
        if outd % 1:
            print('Not compatible dimensions for convolutional layer./n',file=sys.stderr)
            exit(1)
        outd = int(outd)
        # Defining the output shape = (num. filters, output height, output width)
        out_shape = (filter_shape[0],outd,outd)
        # Defining new filter shape = (num. filters, num. layers, filter heights, filter width)
        filter_shape = np.insert(filter_shape,1,in_shape[0])
        
        layer.__init__(self,in_shape,out_shape,filter_shape,filter_shape[0])
        # Padding
        self.npad = ((0,0),(padding,padding),(padding,padding))
        # We now calculate very important class variables for convolutional layers
        # which are the total number of slidings that the filter will perform and others
        self.num_slidings, self.reps, self.sorting_indexes \
                    = gen_indexes(in_shape[-1]+2*padding,filter_shape[-1],stride)

    def forward(self,in_values):
        
        # Padding
        in_values = np.pad(in_values, self.npad, 'constant', constant_values = 0)
        
        out_values \
                = fast3dconv(in_values,self.weights,self.sorting_indexes,
                             self.reps,self.num_slidings)
        return out_values

class maxpool_layer(layer):
    
    def __init__(self,in_shape,stride=3,padding=0):
        
        outd = in_shape[-1]/stride
        
        # Checking dimensional consistency
        if outd % 1:
            print('Not compatible dimensions for convolutional layer./n',file=sys.stderr)
            exit(1)

        outd = int(outd)
        # Defining the output shape
        self.in_shape = in_shape
        self.out_shape = (in_shape[0],outd,outd)
        self.filter_shape = stride
        self.bool_mask = np.zeros(in_shape)
        # Padding
        self.npad = ((0,0),(padding,padding),(padding,padding))
        
    def forward(self,in_values):
        # Padding
        in_values = np.pad(in_values, self.npad, 'constant', constant_values = 0)
        # Divide the input volume slices by NxN using the filter dimension
        in_values = np.array(np.split(in_values,self.out_shape[-1],axis=-1))
        in_values = np.array(np.split(in_values,self.out_shape[-1],axis=-2))
        # Pool the maximum from correct axes and reorder
        out_values = in_values.max(axis=(3,4)).transpose([2,0,1])
        # Build boolean mask for error propagation
        in_values = in_values.transpose([2,0,1,3,4])
        self.bool_mask = (in_values == out_values[...,None,None]).astype(np.int)
        # If we want to reproduce the size of input layer, to be checked
        #bool_mask = fiveDtothreeD(bool_mask)
        return out_values

class ReLU_layer(layer):
    
    def __init__(self,in_shape):
        self.in_shape = in_shape
        self.out_shape = in_shape
        
"""END OF LAYER CLASSES DEFINITION"""