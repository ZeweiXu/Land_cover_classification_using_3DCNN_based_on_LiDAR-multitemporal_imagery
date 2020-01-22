import argparse
import imp
import time
import logging

import numpy as np
from path import Path
import theano
import theano.tensor as T
import lasagne
# import voxnet module and npzloader
import voxnet
from voxnet import npytar
def make_training_functions(cfg, model):
# Define the structural parameters of the 3D deep learning model and initialize tensors
    l_out = model['l_out']
    batch_index = T.iscalar('batch_index')
    X = T.TensorType('float32', [False]*5)('X')
    y = T.TensorType('int32', [False]*1)('y')
    out_shape = lasagne.layers.get_output_shape(l_out)
# separate training batches and define output tensor
    batch_slice = slice(batch_index*cfg['batch_size'], (batch_index+1)*cfg['batch_size'])
    out = lasagne.layers.get_output(l_out, X)
    dout = lasagne.layers.get_output(l_out, X, deterministic=True)
# Read in tensor parameters and initialize training parameters
    params = lasagne.layers.get_all_params(l_out)
    l2_norm = lasagne.regularization.regularize_network_params(l_out,
            lasagne.regularization.l2)
    if isinstance(cfg['learning_rate'], dict):
        learning_rate = theano.shared(np.float32(cfg['learning_rate'][0]))
    else:
        learning_rate = theano.shared(np.float32(cfg['learning_rate']))
# Construct fully-connected layer, softmax function and calculate the final prediciton label
    softmax_out = T.nnet.softmax( out )
    loss = T.cast(T.mean(T.nnet.categorical_crossentropy(softmax_out, y)), 'float32')
    pred = T.argmax( dout, axis=1 )
    error_rate = T.cast( T.mean( T.neq(pred, y) ), 'float32' )
# Calculate loss after normalization
    reg_loss = loss + cfg['reg']*l2_norm
    updates = lasagne.updates.momentum(reg_loss, params, learning_rate, cfg['momentum'])
# Create enmpty array
    X_shared = lasagne.utils.shared_empty(5, dtype='float32')
    y_shared = lasagne.utils.shared_empty(1, dtype='float32')
# Get output label
    dout_fn = theano.function([X], dout)
    pred_fn = theano.function([X], pred)
# Store intermediate training process information
    update_iter = theano.function([batch_index], reg_loss,
            updates=updates, givens={
            X: X_shared[batch_slice],
            y: T.cast( y_shared[batch_slice], 'int32'),
        })
# Use dictionary to store input variables
    error_rate_fn = theano.function([batch_index], error_rate, givens={
            X: X_shared[batch_slice],
            y: T.cast( y_shared[batch_slice], 'int32'),
        })
    tfuncs = {'update_iter':update_iter,
             'error_rate':error_rate_fn,
             'dout' : dout_fn,
             'pred' : pred_fn,
            }
    tvars = {'X' : X,
             'y' : y,
             'X_shared' : X_shared,
             'y_shared' : y_shared,
             'batch_slice' : batch_slice,
             'batch_index' : batch_index,
             'learning_rate' : learning_rate,
            }
    return tfuncs, tvars

def jitter_chunk(src, cfg):
# ramdomly reshuffle samples 
    dst = src.copy()
    if np.random.binomial(1, .2):
        dst[:, :, ::-1, :, :] = dst
    if np.random.binomial(1, .2):
        dst[:, :, :, ::-1, :] = dst
    max_ij = cfg['max_jitter_ij']
    max_k = cfg['max_jitter_k']
    shift_ijk = [np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_k, max_k)]
    for axis, shift in enumerate(shift_ijk):
        if shift != 0:
            dst = np.roll(dst, shift, axis+2)
    return dst

def data_loader(cfg, fname,epoch):
# Read in .npz data and prepare for training
    dims = cfg['dims']
    chunk_size = cfg['batch_size']*cfg['batches_per_chunk']
    xc = np.zeros((chunk_size, cfg['n_channels'],)+dims, dtype=np.float32)
    reader = npytar.NpyTarReader(fname)
    yc = []
    for ix, (x, name) in enumerate(reader):
        cix = ix % chunk_size
        xc[cix] = x.astype(np.float32)
        yc.append(int(name.split('.')[0])-1)
        if len(yc) == chunk_size:
            xc = jitter_chunk(xc, cfg)
            yield (xc, np.asarray(yc, dtype=np.float32))  
            yc = []
            xc.fill(0)

def main(args):
# get parameters from user inputs and printout key steps
    weightsout=args.outdir_weights
    config_module = imp.load_source('config', args.config_path)
# Parameters
    cfg = config_module.cfg
# Import stacked 3D CNN  model 
    model = config_module.get_model()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
    logging.info('Metrics will be saved to {}'.format(args.metrics_fname))
    mlog = voxnet.metrics_logging.MetricsLogger(args.metrics_fname, reinitialize=True)

    logging.info('Compiling theano functions...')
    tfuncs, tvars = make_training_functions(cfg, model)

    logging.info('Training...')
    itr = 0
    last_checkpoint_itr = 0
# Training epoches
    for epoch in xrange(cfg['max_epochs']):
        loader = (data_loader(cfg, args.training_fname,epoch))
        counttt=0
# Loading data from generator
        for x_shared, y_shared in loader:           
            counttt=counttt+1
            num_batches = len(x_shared)//cfg['batch_size']
            tvars['X_shared'].set_value(x_shared, borrow=True)
            tvars['y_shared'].set_value(y_shared, borrow=True)
            lvs,accs = [],[]
            for bi in xrange(num_batches):              
                lv = tfuncs['update_iter'](bi)
                lvs.append(lv)
                acc = 1.0-tfuncs['error_rate'](bi)
                accs.append(acc)
                itr += 1
            loss, acc = float(np.mean(lvs)), float(np.mean(acc))
            logging.info('epoch: {}, itr: {}, loss: {}, acc: {}'.format(epoch, itr, loss, acc))
            mlog.log(epoch=epoch, itr=itr, loss=loss, acc=acc)
# store the parameter providing the highest trainig accuracy
            if counttt==1:
                temacc=acc
                temloss=loss
                temepoch=epoch
                temitr=itr
            else:
                if acc>temacc:
                    temacc=acc
                    temloss=loss
                    temepoch=epoch
                    temitr=itr
            if isinstance(cfg['learning_rate'], dict) and itr > 0:
                keys = sorted(cfg['learning_rate'].keys())
                new_lr = cfg['learning_rate'][keys[np.searchsorted(keys, itr)-1]]
                lr = np.float32(tvars['learning_rate'].get_value())
                if not np.allclose(lr, new_lr):
                    logging.info('decreasing learning rate from {} to {}'.format(lr, new_lr))
                    tvars['learning_rate'].set_value(np.float32(new_lr))
# Checkpoint saviing
            if itr-last_checkpoint_itr > cfg['checkpoint_every_nth']:
                voxnet.checkpoints.save_weights(weightsout+'weights_augI.npz', model['l_out'],
                                                {'itr': itr, 'ts': time.time()})
                last_checkpoint_itr = itr
# Save the weights produce the highest training accuracy
    voxnet.checkpoints.save_weights(weightsout+'weights_bestright_'+str(temacc)+'_'+str(temloss)+'_'+str(temepoch)+'_'+str(temitr)+'.npz', model['l_out'],
                                    {'itr': itr, 'ts': time.time()})

if __name__=='__main__':
# User input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=Path, help='config .py file')
    parser.add_argument('training_fname', type=Path, help='training .tar file')
    parser.add_argument('--metrics-fname', type=Path, default='metrics_augI.jsonl', help='name of metrics file')
    parser.add_argument('--outdir_weights', type=Path, default='/home/cc/3DCNN/data/', help='directory to save weights file')
    args = parser.parse_args()
    main(args)

