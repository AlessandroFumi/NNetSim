import numpy as np

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

"""LAYER CLASSES DEFINITION"""
class layer(object):

    def __init__(self,X_shape,out_shape,W_shape,b_shape):
        #   Maybe we should just save the weights, biases and dimensions
        self.X_shape = X_shape
        self.out_shape = out_shape
        self.W_shape = W_shape
        self.b_shape = b_shape
        #   Initializing weights and biases to random values
        self.W = np.random.randn(np.prod(W_shape)).reshape(W_shape)
        self.b = np.random.randn(b_shape)
        #   Initializing values and errors to 0
        self.X = np.zeros(self.X_shape)
        self.dX = np.zeros(self.X_shape)
        self.out = np.zeros(self.out_shape)
        self.dout = np.zeros(self.out_shape)
    
    def pad_input(self,X):
        # Padding
        npad = ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding))
        X = np.pad(X, npad, 'constant', constant_values = 0)
        return X

    def reset(self):
        self.X = np.zeros(self.X_shape)
        self.dX = np.zeros(self.X_shape)
        self.out = np.zeros(self.out_shape)
        self.dout = np.zeros(self.out_shape)
        return
        
    def randomize(self):
        self.W = np.random.randn(np.prod(self.W_shape)).reshape(self.W_shape)
        self.b = np.random.randn(self.b_shape)
        return
        
class fc_layer(layer):
    def __init__(self,X_shape,out_shape):
        # Input shape = (num. input neurons)
        # Output shape = (num. output neurons)
        # Number of images per batch = (n_images)
        X_shape = (X_shape[0],np.prod(X_shape[1:]))
        out_shape = (X_shape[0],np.prod(out_shape))
        W_shape = (X_shape[-1],out_shape[-1])
        layer.__init__(self,X_shape,out_shape,W_shape,out_shape[-1])

    def forward(self,X):
        self.X = X
        # We use the dot product to calculate the net input of every neuron,
        # and subsequently add the bias vector
        out = np.dot(X.reshape(self.X_shape),self.W) + self.b
        return out

    def backward(self,dout):
        db = dout
        dX = np.dot(dout, self.W.transpose())
        dW = np.einsum('ij,ik->ijk',self.X,dout)
        return dX,dW,db

class conv_layer(layer):
    def __init__(self,X_shape,W_shape,stride=1,padding=0):
        # Input shape = (input depth, input height, input width)
        # Filter shape = (num. filters, filter height, filter width)
        # Additional properties for convolutional/maxpool layers
        self.padding = padding
        self.stride = stride
        #   Calculating output shape
        outdim = (X_shape[-1]+2*padding-W_shape[-1])/stride+1
        # Checking dimensional consistency
        assert (outdim % 1 == 0)
        outdim = int(outdim)
        # Defining the output shape = (num. filters, output height, output width)
        out_shape = (X_shape[0],W_shape[0],outdim,outdim)
        # Defining new filter shape = (num. filters, input depth, filter heights, filter width)
        W_shape = np.insert(W_shape,1,X_shape[1])
        #Applying parent init method
        layer.__init__(self,X_shape,out_shape,W_shape,W_shape[0])

    def forward(self, X):
        #   Check
        assert (X.shape == self.X_shape)
        n_filters, d_filter, h_filter, w_filter = self.W.shape
        n_x, d_x, h_x, w_x = X.shape
        h_out, w_out = self.out_shape[-2:]

        X_col = im2col_indices(X, h_filter, w_filter,self.padding,self.stride)
        W_col = self.W.reshape(n_filters, -1)
        
        out = W_col @ X_col + self.b[...,None]
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        # Memorizing the last input values is useful for the backward step
        self.X = X
        self.X_col = X_col
        return out

    def backward(self, dout):
        X_shape, W, stride, padding, X_col = \
        self.X_shape, self.W, self.stride, self.padding, self.X_col 

        n_filter, d_filter, h_filter, w_filter = self.W_shape

        db = np.sum(dout, axis=(0, 2, 3))
        db = db.reshape(n_filter, -1)

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dW = dout_reshaped @ X_col.T
        dW = dW.reshape(W.shape)

        W = W.reshape(n_filter, -1)
        dX_col = W.T @ dout_reshaped
        dX = col2im_indices(dX_col, X_shape, h_filter, w_filter, padding=padding, stride=stride)
        return dX, dW, db

class maxpool_layer(layer):
    
    def __init__(self,X_shape,stride=3,padding=0):
        # Input shape = (input depth, input height, input width)
        # Filter shape = (stride)
        #   Additional properties for convolutional/maxpool layers
        self.padding = padding
        self.stride = stride
        # Defining the output shape = (input depth, output height, output width)
        outdim = ( X_shape[-1] + 2*padding ) / stride
        assert (outdim % 1 == 0)
        outdim = int(outdim)
        out_shape = (X_shape[:1],outdim,outdim)
        layer.__init__(self,X_shape,out_shape,0,0)
        del self.W, self.b, self.W_shape, self.b_shape
        # Now the boolean mask, which is important for backpropagation
        self.bool_mask = np.zeros(X_shape)

    def forward(self,X):
        # Padding
        X = self.pad_input(X)
        # Divide the input volume slices by NxN using the filter dimension
        X = np.array(np.split(X,self.out_shape[-1],axis=-1))
        X = np.array(np.split(X,self.out_shape[-1],axis=-2))
        # Pool the maximum from correct axes and reorder
        out = X.max(axis=(-2,-1)).transpose([2,3,0,1])
        # Build boolean mask for error propagation
        X = X.transpose([2,3,0,1,4,5])
        self.bool_mask = (X == out[...,None,None]).astype(np.int)
        # If we want to reproduce the size of input layer, to be checked
#        self.bool_mask = self.bool_mask.transpose(0,1,2,4,3,5).reshape(self.X_shape)
        return out
    
    def backward(self,dout):
        np.einsum('...ij,...->...ij',self.bool_mask,dout)
        dX = self.bool_mask = self.bool_mask.transpose(0,1,2,4,3,5).reshape(self.X_shape)
        return dX

class ReLU_layer(layer):
    def __init__(self,X_shape):
        layer.__init__(self,X_shape,X_shape,0,0)
        del self.W, self.b, self.W_shape, self.b_shape
        
    def forward(self, X):
        self.X = X
        out = X.clip(min=0)
        return out
    
    def backward(self, dout):
        # Normalization + 'clip' routine that simulates the ReLU behaviour
        dX = np.multiply(dout,(self.X > 0.0).astype(np.float))
        return dX
"""END OF LAYER CLASSES DEFINITION"""

"""NEURALNET CLASS DEFINITION"""
class NeuralNet(object):
    def __init__(self,layerList,paramList,batch_size,X_max):
        self.batch_size = batch_size
        self.X_max = X_max
        self.layers = []
        self.layerCount = 0
        for i in layerList:
            self.layers.append(layerList[self.layerCount](*paramList[self.layerCount]))
            self.layerCount += self.layerCount
        self.num_classes = self.layers[-1].out_shape

    def gen_gtruth(self,labels):
    #   This might be modified for more complex encodings
        assert ((labels < self.num_classes).all())
        assert (len(labels) == self.batch_size)
        
        gtruth = np.zeros((self.batch_size,self.num_classes))
        gtruth[np.arange(self.batch_size),labels] = self.X_max
        return gtruth
    
    def train(self,training_set,learning_rate):
        # Initializing and checking some variables
        batch_size = self.batch_size
        assert (len(training_set) % batch_size == 0)
        # Extracting labels and images, assuming that the training set is a list of tuples
        labels, images = zip(*training_set)
        # Creating batches (don't know how it will work when batch_size = 1)
        batches = [np.array(images[i:i + batch_size]) for i in range(0, len(images), batch_size)]
        labels = [np.array(labels[i:i + batch_size]) for i in range(0, len(labels), batch_size)]
        
        # Actual computation
        for e, i in enumerate(batches):
            for j in self.layers:
                i = j.forward(i)
            # Computing output error
            i = self.gen_gtruth(labels[e]) - i
            # Updating accuracy
            pass

            for j in reversed(self.layers):
                i = j.backward(i)

    def verification(self):
        pass
        
    def evaluate(self,test_set):
        pass
    
    def saveStatus(self):
        pass
    
    def loadStatus(self):
        pass
"""END OF NEURALNET CLASS DEFINITION"""
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        