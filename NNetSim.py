import numpy as np
from scipy.special import expit as logistic

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

""" END OF BASIC BUILDING BLOCKS OF EVERY LAYER """
class neurons(object):
    def __init__(self,X_shape,out_shape, slope = 1):
        #   Maybe we should just save the weights, biases and dimensions
        self.X_shape = X_shape
        self.out_shape = out_shape
        #   Initializing values and errors to 0
        self.X = np.zeros(self.X_shape)
        self.out = np.zeros(self.out_shape)
        self.dout = np.zeros(self.out_shape)
        #   Next line is used for activation neurons, not implemented yet
        self.slope = slope

    def loadvalues(self,X):
        self.X = X.reshape(self.X_shape)
    
    def loaderrors(self,dout):
        self.dout = dout.reshape(self.out_shape)

    def pad(self):
        # Padding
        npad = ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding))
        return np.pad(self.X, npad, 'constant', constant_values = 0)

    def reset(self):
        self.X.fill(0)
        self.out.fill(0)
        self.dout.fill(0)

class synapses(object):
    
    def __init__(self,W_shape,b_shape,learning_rate = 10):
        #   Maybe we should just save the weights, biases and dimensions
        self.W_shape = W_shape
        self.b_shape = b_shape
        #   Initializing weights and biases to random values
        self.W = np.random.randn(*W_shape)
        self.b = np.random.randn(b_shape)
        # Let's put the learning rate here for now
        self.eta = learning_rate
        
    def update(self,dW,db):
#        print('Weight sum: %f' % np.sum(self.W))
#        print('Delta weight sum: %f' % np.sum(dW))
        self.W += self.eta*dW
        self.b += self.eta*db
        

    def randomize(self):
        self.W = np.random.randn(*self.W_shape)
        self.b = np.random.randn(self.b_shape)
        
    def save(self):
        pass
    
    def load(self):
        pass
""" END OF BASIC BUILDING BLOCKS OF EVERY LAYER """

"""LAYER CLASSES DEFINITION"""
class ReLU_layer(neurons):
    def __init__(self,X_shape):
        neurons.__init__(self,X_shape,X_shape)
        
    def forward(self):
        return self.X.clip(min=0)
    
    def backward(self):
        return np.multiply(self.dout,(self.X > 0.0).astype(np.float))

class PWL_layer(neurons):
    def __init__(self,X_shape):
        neurons.__init__(self,X_shape,X_shape)
        
    def forward(self):
        return self.X.clip(min=0,max=1)
    
    def backward(self):
        return np.multiply(self.dout,(np.logical_and(self.X > 0.0, self.X < 1.0) ).astype(np.float))

class sigmoid_layer(neurons):
    def __init__(self,X_shape):
        neurons.__init__(self,X_shape,X_shape)
        
    def forward(self):
        self.out = logistic(self.X)
        return self.out
    
    def backward(self):
        return np.multiply(self.dout,self.out * (1 - self.out))

class tanh_layer(neurons):
    def __init__(self,X_shape):
        neurons.__init__(self,X_shape,X_shape)
        
    def forward(self):
        self.out = 0.5*np.tanh(self.X) + 0.5
        return self.out
    
    def backward(self):
        return np.multiply(self.dout,(1 - np.power(self.out,2)))

class norm_layer(neurons):
    def __init__(self,X_shape):
        neurons.__init__(self,X_shape,X_shape)
    
    def forward(self):
        return np.divide(
                    self.X,
                    np.max(self.X,axis = -1)[...,None]
                    )
    
    def backward(self):
        # This implementation is not correct, as we should take into account the compound derivative
        # but this layer is supposed to be used just as input, so the backward is not important
        return np.multiply(
                        self.dout,
                        np.max(self.X,axis = -1)[...,None]
                        )

class softmax_layer(neurons):
    def __init__(self,X_shape):
        neurons.__init__(self,X_shape,X_shape)
        
    def forward(self):
        self.out = np.divide(
                    np.exp(self.X),
                    np.sum(np.exp(self.X),axis = -1)[...,None]
                   )
        return self.out
    
    def backward(self):
        jacobian = - np.einsum('ij,ik->ijk',self.out,self.out) \
                   + np.einsum('ij,...j->ij...',self.out,np.eye(self.out_shape[-1]))
        # Be careful because in the previous line i put the temporary result in dout
        return np.einsum('ijk,ik ->ij',jacobian,self.dout)


class fc_layer(neurons,synapses):
    def __init__(self,X_shape,out_shape):
        # Input shape = (num. input neurons)
        # Output shape = (num. output neurons)
        # Number of images per batch = (n_images)
        X_shape = (X_shape[0],np.prod(X_shape[1:]))
        out_shape = (X_shape[0],np.prod(out_shape))
        W_shape = (X_shape[-1],out_shape[-1])
        # Actual initialization
        neurons.__init__(self,X_shape,out_shape)
        synapses.__init__(self,W_shape,out_shape[-1])

    def forward(self):
        # We use the dot product to calculate the net input of every neuron,
        # and subsequently add the bias vector
        return np.dot(self.X,self.W) + self.b

    def backward(self):
        self.update(np.sum(np.einsum('ij,ik->ijk',self.X,self.dout),axis=0),
                    np.sum(self.dout,axis=0))
        return np.dot(self.dout, self.W.transpose())

class conv_layer(neurons,synapses):
    def __init__(self,X_shape,W_shape,stride=1,padding=0):
        # Input shape = (input depth, input height, input width)
        # Filter shape = (num. filters, filter height, filter width)
        assert(len(W_shape)==3)
        assert(len(X_shape)==4)
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
        neurons.__init__(self,X_shape,out_shape)
        synapses.__init__(self,W_shape,W_shape[0])

    def forward(self):
        n_filters, d_filter, h_filter, w_filter = self.W.shape
        n_x, d_x, h_x, w_x = self.X_shape
        h_out, w_out = self.out_shape[-2:]

        X_col = im2col_indices(self.X, h_filter, w_filter,self.padding,self.stride)
        W_col = self.W.reshape(n_filters, -1)
        
        out = W_col @ X_col + self.b[...,None]
        out = out.reshape(n_filters, h_out, w_out, n_x)
        # Memorizing the last input values is useful for the backward step
        self.X_col = X_col
        return out.transpose(3, 0, 1, 2)

    def backward(self):
        X_shape, W, stride, padding, X_col = \
        self.X_shape, self.W, self.stride, self.padding, self.X_col 

        n_filter, d_filter, h_filter, w_filter = self.W_shape

        # Calculate db and update
        db = np.sum(self.dout, axis=(0, 2, 3))
        # Calculate dW and update
        dout_reshaped = self.dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dW = dout_reshaped @ X_col.T
        # Weight update
        self.update(dW.reshape(self.W_shape),db.reshape(n_filter))
        # I don't know if flipping the weights it's correct
        W = np.flip(W,axis=-1)
        W = np.flip(W,axis=-2)
        W = W.reshape(n_filter, -1)

        dX_col = W.T @ dout_reshaped
        return col2im_indices(dX_col, X_shape, h_filter, w_filter, padding=padding, stride=stride)

class maxpool_layer(neurons):
    
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
        out_shape = X_shape[:2]+(outdim,outdim)
        neurons.__init__(self,X_shape,out_shape)
        # Now the boolean mask, which is important for backpropagation
        self.bool_mask = np.zeros(X_shape)

    def forward(self):
        # Padding
        X = self.pad()
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
    
    def backward(self):
        return (self.bool_mask*self.dout[...,None,None]).transpose([0,1,2,4,3,5]).reshape(self.X_shape)
    
    def reshaped_mask(self):
        return self.bool_mask.transpose([0,1,2,4,3,5]).reshape(self.X_shape)

"""END OF LAYER CLASSES DEFINITION"""

"""NEURALNET CLASS DEFINITION"""
#class NeuralNet(object):
#    # Probably the best way to parse this guy is:
#        #   1) Interactive session (that saves a dictioary)
#        #   2) Python dictionary
#    def __init__(self,kwargs):
#        for key, value in kwargs.items():
#            setattr(self, key, value)
#        # Checking correctness of parameters:
#        assert (len(self.layerList) == len(self.layerParamList))
#        self.layers = []
#        self.layerCount = 0
#        
#        """ This part of code can be improved """
#        for e,i in enumerate(self.layerList):
#            if e == 0:
#                self.layers.append(self.layerList[e](*self.layerParamList[e]))
#            else:
#                self.layers.append(self.layerList[e](self.layers[e-1].out_shape,
#                                   *self.layerParamList[e]))
#            print('Layer %d successfully created.' % e)
#            self.layerCount += self.layerCount
#        """ End of part of code can be improved """
#        #   Additional (unnecessary) information
#        self.input_layer= self.layers[0]
#        self.batch_size = self.input_layer.X_shape[0]
#        self.output_layer = self.layers[-1]
#        self.num_classes = self.output_layer.out_shape[-1]
#
#    def gen_gtruth(self,labels):
#        # This might be modified for more complex encodings
#        assert ((labels < self.num_classes).all())
#        assert ( np.size(labels, axis = 0) == self.batch_size)
#
#        gtruth = np.zeros((self.batch_size,self.num_classes))
#        gtruth[np.arange(self.batch_size),labels] = 1
#        return gtruth
#
#    def train(self,training_set):
#        # Some checking
##        assert ( np.max(labels) + 1 == self.num_classes )
#        # Initializing and checking some variables
#        batch_size = self.batch_size
#        samples = 0
#        correct = 0
#        assert (len(training_set) % batch_size == 0)
#        # Creating batches (don't know how it will work when batch_size = 1)
#        batches = [training_set[i:i + self.batch_size] for i in range(0, len(training_set), self.batch_size)]
#        
#        # Actual computation
#        for i in batches:
#            labels, i = zip(*i)
#            labels = np.array(labels)
#            i = np.array(i)
#            
#            for j in self.layers:
#                j.loadvalues(i)
#                i = j.forward()
#
#            # Checking correctness: checking argmax of last neurons vs number of classes
#            # and then summing all the times our neurons were correct
#            samples += self.batch_size
#            correct += np.sum(labels == np.argmax(i,axis = -1))
#            # Computing output error
#            i = self.gen_gtruth(labels) - i
##            print('Cost function: %f' % np.sum(np.power(self.gen_gtruth(labels[e]) - i,2)))
#
#            for j in reversed(self.layers):
#                j.loaderrors(i)
#                i = j.backward()
#            
#            #Print
#            
#            print('Accuracy = %f' % float(correct*100/samples))
#            
#        return 0
#        
#    def verification(self):
#        pass
#        
#    def evaluate(self,test_set):
#        pass
#    
#    def saveStatus(self):
#        pass
#    
#    def loadStatus(self):
#        pass
#    
class NeuralNet(object):
    '''
    This class is a supposedly easy to use wrapper that should control
    the information flow amongst the layer objects present in the neural net.
    The main purpose of the NeuralNet class is to correctly transfer values and
    gradient between layers. The layer object are contained into the list:
    self.layers.
    '''
    def __init__(self,**kwargs):
        '''
        So far I don't think we need something specific for the neural net class
        so we'll just leave the classic dictionary initialization that should
        work as following:
            
            params = {
                    'param1' : 3,
                    'param2' : [1,8,4.5]
                    }
            
            #   Using external dictionaries, note the double star notation
            #   to 'unroll' the dictionary into keyword arguments
            x = NeuralNet(**params)
            x.param1 # Should print 3
            x.param2 # Should print [1,8,4.5]
            
            #   Using directly keyword arguments
            x = NeuralNet(param1 = 3, param2 = [1,8,4.5])
            x.param1 # Should print 3
            x.param2 # Should print [1,8,4.5]
            
            The two methods are equivalent.
        '''
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        '''
        We initialize the layer list, as we'll just use the .append and .pop
        methods to modify it during the execution of the code
        '''
        self.layers = []

    def add_layer(self, layerobj, *args):
        print(*args)
        if args:
            '''
            If args means that the user specified the layer (layerobj) and the 
            arguments separately. This is a 'safer' automatic procedure to add a 
            layer, as the code will automatically make it consistent with the 
            already added layers. The code might look something like this:
        
                x = NeuralNet()        # So far the init method does not require args
                x.addlayer( fc_layer, (10,784) , (10,))
            '''
            if not self.layers:
                self.layers.append( layerobj ( *args ) )
            else:
                self.layers.append( layerobj ( self.layers[-1].out_shape, *args ) )

        else:
            '''
            iF the user did not specify 'args' separately, it means that layerobj
            is already a layer class parsed with the proper input parameters, 
            for example, in this case the code could look something like this:
                
                x = NeuralNet()        # So far the init method does not require args
                x.addlayer( fc_layer( (10,784) , (10,) ) )
                
            This can be a little more risky
            '''
            self.layers.append( layerobj )
        
        '''
        To print out the name of the layer we could have directly used layerobj.__name__ in
        the first case of the if statement, to be more versatile we will call the 
        layerobj().__class__.__name__ method instead (which is more general)
        '''
        print('Successfully created %s as layer number %d' % (self.layers[-1].__class__.__name__, len(self.layers)))
#        print('Parameters for layer %d are: \n Input_shape = %s \n Output_shape = %s \n' 
#              % ( len(self.layers), self.layers[-1].X_shape, self.layers[-1].out_shape ) )

    def remove_layer(self, indexes):
        for i in indexes:
            '''
            Every layer inherited a method from the 'neuron' class called self.loadvalues
            that automatically check if the inputs the layer is receiving
            are compatible with the layer specification. It also needs to be reminded
            that the self.forward method for layer objects returns the output computed with some
            inputs which are already stored in the self.X variable, which usually is 
            initialized to all zeroes. The self.loadvalues method throws an assertionError
            in case the shape of the incoming input and the one of the self.X variable
            are not broadcastable.
            '''
            if i != 0:
                self.layers[i+1].loadvalues(self.layers[i-1].forward())
        self.layers.pop(indexes)

        '''
        In the future we could add a method to automatically change the other layers
        to adapt to the removal of a middle layer, as an example:
            
            INITIAL NETWORK:
            FC (784,250) --> FC (250,125) --> FC (125,10)
            
            NETWORK AFTER REMOVAL OF HIDDEN LAYER:
            FC (784,250) --> FC (250,10)
        '''
        
    def gen_gtruth(self,labels):
        '''
        Assuming one-hot encoding in the output layer, and labels in the form
        of a vector of integer numbers (with values ranging from 0 to N_class - 1)
        
        '''
        assert ((labels < self.num_classes).all())
        assert ( np.size(labels, axis = 0) == self.batch_size)

        gtruth = np.zeros((self.batch_size,self.num_classes))
        gtruth[np.arange(self.batch_size),labels] = 1
        return gtruth
        
    def train(self,training_set):
        '''
        Here comes the fun part
        '''
        '''
        Additional, maybe unnecessary, information
        '''
        self.input_layer= self.layers[0]
        self.batch_size = self.input_layer.X_shape[0]
        self.output_layer = self.layers[-1]
        self.num_classes = self.output_layer.out_shape[-1]
        # Initializing and checking some variables
        samples = 0
        correct = 0
        assert (len(training_set) % self.batch_size == 0)
        # Creating batches (don't know how it will work when batch_size = 1)
        batches = [training_set[i:i + self.batch_size] for i in range(0, len(training_set), self.batch_size)]
        # Actual computation
        for i in batches:
            labels, i = zip(*i)
            labels = np.array(labels)
            i = np.array(i)
            
            for j in self.layers:
                j.loadvalues(i)
                i = j.forward()

            # Checking correctness: checking argmax of last neurons vs number of classes
            # and then summing all the times our neurons were correct
            samples += self.batch_size
            correct += np.sum(labels == np.argmax(i,axis = -1))
            # Computing output error
            i = self.gen_gtruth(labels) - i
#            print('Cost function: %f' % np.sum(np.power(self.gen_gtruth(labels[e]) - i,2)))

            for j in reversed(self.layers):
                j.loaderrors(i)
                i = j.backward()
            
            #Print
            
            print('Accuracy = %f' % float(correct*100/samples))

    def verification(self):
        pass
        
    def evaluate(self,test_set):
        pass
    
    def saveStatus(self):
        pass
    
    def loadStatus(self):
        pass
"""END OF NEURALNET CLASS DEFINITION"""