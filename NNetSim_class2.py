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

def preconv_expand(in_values,weights,sorting_indexes,reps,num_slidings):
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
    return in_values,weights

def fast3dconv(in_values,weights,sorting_indexes,reps,num_slidings):
    
    in_values , weights = preconv_expand(in_values,weights,sorting_indexes,reps,num_slidings)
    # Einstein product: convolution without biasing
    out_values = np.einsum('...ijk,...ijk->...', in_values , weights)
    return out_values
"""END OF FUNCTIONS FOR EFFICIENT 2D CONVOLUTION"""

"""FUNCTIONS FOR EFFICIENT 2D CONVOLUTION, im2col bases"""
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = int((H + 2 * padding - field_height) / stride + 1)
  out_width = int((W + 2 * padding - field_width) / stride + 1)

  i0 = np.repeat(np.arange(field_height), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
  """ An implementation of col2im based on fancy indexing and np.add.at """
  N, C, H, W = x_shape
  H_padded, W_padded = H + 2 * padding, W + 2 * padding
  x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
  k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
  cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)
  np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
  if padding == 0:
    return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]
"""FUNCTIONS FOR EFFICIENT 2D CONVOLUTION, im2col bases"""
def conv_forward(X, W, b, stride=1, padding=1):
    cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = W_col @ X_col + b
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, W, b, stride, padding, X_col)

    return out, cache

def conv_backward(dout, cache):
    X, W, b, stride, padding, X_col = cache
    n_filter, d_filter, h_filter, w_filter = W.shape

    db = np.sum(dout, axis=(0, 2, 3))
    db = db.reshape(n_filter, -1)

    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    dW = dout_reshaped @ X_col.T
    dW = dW.reshape(W.shape)

    W_reshape = W.reshape(n_filter, -1)
    dX_col = W_reshape.T @ dout_reshaped
    dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

    return dX, dW, db
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
                self.num_slidings, self.reps, self.sorting_indexes = \
                                gen_indexes(self.padded_in_shape[-1],weight_shape[-1],stride)

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
        return
        
    def layer_retrain(self):
        self.weights = np.random.randn(np.prod(self.weight_shape)).reshape(self.weight_shape)
        self.biases = np.random.randn(self.bias_shape)
        return
        
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

    def preconv_expand(self, in_values):
        # Padding
        in_values = self.pad_input(in_values)
        # Replicate and ordering rows and columns
        in_values = np.repeat(in_values,self.reps,axis=-1)
        in_values = in_values[...,self.sorting_indexes]
        in_values= np.repeat(in_values,self.reps,axis=-2)
        in_values = in_values[...,self.sorting_indexes,:]
        # Splitting in equal parts
        in_values= np.array(np.split(in_values,self.num_slidings,axis=-1))
        in_values= np.array(np.split(in_values,self.num_slidings,axis=-2))
    
        #Replicating weights
        weights = np.tile(self.weights,(self.num_slidings,self.num_slidings))
        weights = np.array(np.split(weights,self.num_slidings,axis=-1))
        weights = np.array(np.split(weights,self.num_slidings,axis=-2))
        #Here we have the assumption that weights are a 4D tensor turned 6D
        weights = weights.transpose([2,0,1,3,4,5])
        return in_values,weights

    def fast3dconv(self,in_values):
        
        in_values , weights = self.preconv_expand(in_values)
        # Einstein product: convolution without biasing
        out_values = np.einsum('...ijk,...ijk->...', in_values , weights)
        return out_values
    
    def forward(self,in_values):
        out_values \
                = self.fast3dconv(in_values) + self.biases[...,None,None]
        return out_values

#    def backward(self,out_err):
#        weights = np.flip(self.weights,axis=-1)
#        weights = np.flip(weights,axis=-2)
#        # The backpropagation of a convolutional layer is still a convolution
#        in_err \
#                = fast3dconv(out_err,weights,self.sorting_indexes,
#                             self.reps,self.num_slidings)
#        return in_err

    def backward(self,out_err):
        out_err = out_err[...,None,None,None]
        in_err, weights = self.preconv_expand(self.in_err)
        in_err = np.einsum('i...,i...->...',out_err,self.weights)
        return in_err

    def test_conv(self):
        test_input = np.array([1,2,0,2,1,
                                2,1,0,1,0,
                                1,0,0,1,2,
                                1,0,0,2,0,
                                0,1,0,0,2,
                                2,2,1,1,2,
                                1,1,1,1,1,
                                1,1,1,0,1,
                                2,2,2,0,0,
                                2,0,1,1,2,
                                2,1,1,2,2,
                                0,0,0,1,0,
                                0,2,2,2,1,
                                1,0,2,0,2,
                                2,2,2,2,2]).reshape(3,5,5)
        test_weights = np.array([-1, 1, 0,
                                  0, 0, -1,
                                  0, 0, 1,
                                  0, 0, 1,
                                  1, 0, 1,
                                  1, 0, 1,
                                  1, -1, 0,
                                  -1, -1, 0,
                                  1, 0, 0,
                                  -1, 0, -1,
                                  0, 0, 0,
                                  1, 1, -1,
                                  1, 1, 1,
                                  0, 0, 0,
                                  0, 0, 1,
                                  1, -1, 0,
                                  0, -1, 1,
                                  -1, -1, -1]).reshape(2,3,3,3)

        test_biases = np.array([1, 0])

        test_output = np.array([[[ 1,  3,  0],
                                  [ 7,  1, -2],
                                  [ 0, -4, -6]],
                                 [[ 1,  1, -2],
                                  [ 5, -3,  1],
                                  [ 3,  0, -6]]]).reshape(2,3,3)

        padding = 1
        stride = 2
        layer.__init__(self,test_input.shape,None,(test_weights.shape[0],test_weights.shape[2],test_weights.shape[3]),test_weights.shape[0],stride,padding)

        self.weights = test_weights
        self.biases = test_biases

        if (test_output == self.forward(test_input)).all():
            print('Forward Convolution OK\n')
        else: print('Forward Convolution not OK\n')

        print(self.backward(test_output))
        return

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