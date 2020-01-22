import numpy as np
import lasagne
import lasagne.layers
import voxnet
# learning rate after how many batches
lr_schedule = { 0: 0.001,
                60000: 0.0001,
                400000: 0.00005,
                600000: 0.00001,
                }
# parameters of the model
cfg = {'batch_size' : 32,
       'learning_rate' : lr_schedule,
       'reg' : 0.001,
       'momentum' : 0.9,
       'dims' : (32,32,52),
       'n_channels' : 1,
       'n_classes' : 7,
       'batches_per_chunk': 64,
       'max_epochs' : 2000,
       'max_jitter_ij' : 2,
       'max_jitter_k' : 2,
       'n_rotations' : 9,
       
       'checkpoint_every_nth' : 4000,
       }
# Create model layers and stack them one by one
def get_model():
    dims, n_channels, n_classes = tuple(cfg['dims']), cfg['n_channels'], cfg['n_classes']
    shape = (None, n_channels)+dims
# Input layer
    l_in = lasagne.layers.InputLayer(shape=shape)
# Convolutional layer #1
    l_conv1 = voxnet.layers.Conv3dMMLayer(
            input_layer = l_in,
            num_filters = 32,
            filter_size = [5,5,5],
            border_mode = 'valid',
            strides = [2,2,2],
            W = voxnet.init.Prelu(),
            nonlinearity = voxnet.activations.leaky_relu_01,
            name =  'conv1'
        )
# Dropout layer with dropping rate of 0.2
    l_drop1 = lasagne.layers.DropoutLayer(
        incoming = l_conv1,
        p = 0.2,
        name = 'drop1'
        )
# Convolutional layer #2
    l_conv2 = voxnet.layers.Conv3dMMLayer(
        input_layer = l_drop1,
        num_filters = 32,
        filter_size = [3,3,3],
        border_mode = 'valid',
        W = voxnet.init.Prelu(),
        nonlinearity = voxnet.activations.leaky_relu_01,
        name = 'conv2'
        )
# Pooling layer with pooling step [2,2,2]
    l_pool2 = voxnet.layers.MaxPool3dLayer(
        input_layer = l_conv2,
        pool_shape = [2,2,2],
        name = 'pool2',
        )
# dropout layer
    l_drop2 = lasagne.layers.DropoutLayer(
        incoming = l_pool2,
        p = 0.3,
        name = 'drop2',
        )
# Dense layer with 128 dimensions
    l_fc1 = lasagne.layers.DenseLayer(
        incoming = l_drop2,
        num_units = 128,
        W = lasagne.init.Normal(std=0.01),
        name =  'fc1'
        )
# dropout layer
    l_drop3 = lasagne.layers.DropoutLayer(
        incoming = l_fc1,
        p = 0.4,
        name = 'drop3',
        )
# Fully connected layer
    l_fc2 = lasagne.layers.DenseLayer(
        incoming = l_drop3,
        num_units = n_classes,
        W = lasagne.init.Normal(std = 0.01),
        nonlinearity = None,
        name = 'fc2'
        )
# output layers
    return {'l_in':l_in, 'l_out':l_fc2,'l_conv1':l_conv1,'l_drop1':l_drop1,'l_conv2':l_conv2,'l_pool2':l_pool2,'l_drop2':l_drop2,'l_128':l_fc1,'l_drop3':l_drop3,'l_fc2':l_fc2}

