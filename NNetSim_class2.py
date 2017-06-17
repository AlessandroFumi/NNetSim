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

    def __init__(self,in_shape,out_shape,weight_shape,bias_shape,stride=0,padding=0):
        #   Additional properties for convolutional/maxpool layers
        if stride > 0: # maxpool or conv layers 
            # Padding
            self.npad = ((0,0),(padding,padding),(padding,padding))
            self.padded_in_shape = tuple(np.add(in_shape,np.sum(self.npad,axis = 1)))
            outd = (self.padded_in_shape[-1]-weight_shape[-1])/stride+1
            # Checking dimensional consistency
            if outd % 1:
                print('Not compatible dimensions for conv or maxpool layer./n',file=sys.stderr)
            outd = int(outd)
            
            if len(weight_shape) <= 2:    #   Max Pool Layer
                # Defining the output shape = (input depth, output height, output width)
                out_shape = (in_shape[0],outd,outd)
            else:   #   Conv Layer
                # Defining the output shape = (num. filters, output height, output width)
                out_shape = (weight_shape[0],outd,outd)
                # Defining new filter shape = (num. filters, input depth, filter heights, filter width)
                weight_shape = np.insert(weight_shape,1,in_shape[0])
                # We now calculate very important class variables for convolutional layers
                # which are the total number of slidings that the filter will perform and others
                # these will be used to perform the customised 3d convolution
                self.num_slidings, self.reps, self.sorting_indexes \
                            = gen_indexes(self.padded_in_shape[-1],weight_shape[-1],stride)

        #   Maybe we should just save the weights, biases and dimensions
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.weight_shape = weight_shape
        self.bias_shape = bias_shape
        self.weights = np.random.randn(np.prod(weight_shape)).reshape(weight_shape)
        self.biases = np.random.randn(bias_shape)
        #   Common properties of every layers
        self.in_values = np.zeros(self.in_shape)
        self.in_err = np.zeros(self.in_shape)
        self.out_values = np.zeros(self.out_shape)
        self.out_err = np.zeros(self.out_shape)

    def pad_input(self,in_values):
        # Padding
        in_values = np.pad(in_values, self.npad, 'constant', constant_values = 0)
        return in_values
        
    def layer_reset(self):
        self.in_values = np.zeros(self.in_shape)
        self.in_err = np.zeros(self.in_shape)
        self.out_values = np.zeros(self.out_shape)
        self.out_err = np.zeros(self.out_shape)
        
class fc_layer(layer):
    def __init__(self,in_shape,out_shape):
        # Input shape = (num. input neurons)
        # Output shape = (num. output neurons)
        layer.__init__(self,np.prod(in_shape),np.prod(out_shape),
                       (np.prod(in_shape),np.prod(out_shape)),np.prod(out_shape))

    def forward(self,in_values):
        # We use the dot product to calculate the net input of every neuron,
        # and subsequently add the bias vector
        out_values = np.add(self.biases.ravel(),
                           np.dot(in_values.ravel(),self.weights))
        return out_values

    def backward(self,out_err):
        in_err = np.dot(out_err, self.weights.transpose())
        return in_err

class conv_layer(layer):
    def __init__(self,in_shape,filter_shape,stride=1,padding=0):
        # Input shape = (input depth, input height, input width)
        # Filter shape = (num. filters, filter height, filter width)
        #Applying parent init method
        layer.__init__(self,in_shape,None,filter_shape,filter_shape[0],stride,padding)
        
    def forward(self,in_values):
        # Padding
        in_values = self.pad_input(in_values)
        out_values \
                = fast3dconv(in_values,self.weights,self.sorting_indexes,
                             self.reps,self.num_slidings)
        return out_values

    def backward(self,out_err):
        weights = np.flip(self.weights,axis=-1)
        weights = np.flip(weights,axis=-2)
        # The backpropagation of a convolutional layer is still a convolution
        in_err \
                = fast3dconv(out_err,weights,self.sorting_indexes,
                             self.reps,self.num_slidings)
        return in_err

class maxpool_layer(layer):
    
    def __init__(self,in_shape,stride=3,padding=0):
        # Input shape = (input depth, input height, input width)
        # Filter shape = (stride)
        layer.__init__(self,in_shape,None,(stride,stride),0,stride,padding)
        # Now the boolean mask, which is important for backpropagation
        self.bool_mask = np.zeros(in_shape)

    def forward(self,in_values):
        # Padding
        in_values = self.pad_input(in_values)
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
        layer.__init__(self,in_shape,in_shape,None,None)
        del self.weights, self.biases, self.weight_shape, self.bias_shape
        
    def forward(self, in_values):
        # Normalization + 'clip' routine that simulates the ReLU behaviour
        out_values = in_values.clip(min=0)
        return out_values
"""END OF LAYER CLASSES DEFINITION"""