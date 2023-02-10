import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import os
import time
import re

import argparse
import pickle

def build_model(n_inputs, n_hidden, n_output, activation='relu', lrate=0.001):
    '''
    Build a simple dense model
    - Adam optimizer
    - MSE loss
    :param n_inputs: Number of input dimensions
    :param n_hidden: Number of units in the hidden layer
    :param n_output: Number of ouptut dimensions
    :param activation: Activation function to be used for hidden and output units
    :param lrate: Learning rate for Adam Optimizer
    '''
    
    # Sequential model
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(layers.InputLayer(input_shape=(n_inputs,)))
    
    # Hidden layers
    for i, n in enumerate(n_hidden):
        model.add(layers.Dense(n, use_bias=True, name='hidden_{:d}'.format(i), activation=activation))
    
    # Output layer
    model.add(layers.Dense(n_output, use_bias=False, name='output', activation='tanh'))
    
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lrate)
    
    # Compile the model (MSE loss)
    model.compile(loss='mse', optimizer=opt)
    
    # Display the network
    print(model.summary())
    
    return model

class ThresholdCallback(tf.keras.callbacks.Callback):
    '''
    A custom callback to stop when target validation loss is achieved
    '''
    def __init__(self, threshold):
        super(ThresholdCallback, self).__init__()
        self.threshold = threshold
    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < self.threshold:
            self.model.stop_training = True

def args2string(args):
    '''
    Translate the current set of arguments
    
    :param args: Command line arguments
    '''
    return "exp_%02d_hidden_%s"%(args.exp, "_".join([str(i) for i in args.hidden]))

def execute_exp(args):
    '''
    Execute a single instance of an experiment.  The details are specified in the args object
    
    :param args: Command line arguments
    '''
    # Run the experiment
    
    # load dataset
    fp = open("hw0_dataset.pkl", "rb")
    foo = pickle.load(fp)
    fp.close()
    ins = foo['ins']
    outs = foo['outs']
    
    # Build model
    model = build_model(ins.shape[1], args.hidden, outs.shape[1], activation=tf.math.sin, lrate=args.lrate)
    
    # Callbacks
    #early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True, min_delta=0.0001)
    callback = ThresholdCallback(0.1)
    
    # Describe arguments
    argstring = args2string(args)
    print("EXPERIMENT: %s"%argstring)
    
    # Only execute if we are 'going'
    if not args.nogo:
        # Training
        
        print("Training...")
        
        # Note: faking validation data set
        history = model.fit(x=ins, y=outs, epochs=args.epochs, verbose=(args.verbose >= 2), validation_data=(ins, outs), callbacks=[callback])
        
        print("Done Training")
        
        # Save the training history
        with open('results/hw0_results_{}.pkl'.format(argstring), 'wb') as fp:
            pickle.dump(history.history, fp)
            pickle.dump(args, fp)

def display_learning_curve(fname):
    '''
    Display the learning curve that is stored in fname
    
    :param fname: Results file to load and dipslay
    '''
    
    # Load the history file and display it
    fp = open(fname, "rb")
    history = pickle.load(fp)
    fp.close()
    
    # Display
    plt.plot(history['loss'])
    plt.ylabel('MSE')
    plt.xlabel('epochs')

def display_learning_curve_set(dir, base):
    '''
    Plot the learning curves for a set of results
    
    :param base: Directory containing a set of results files
    '''
    # Find the list of files in the local directory that match base_[\d]+.pkl
    files = [f for f in os.listdir(dir) if re.match(r'%s.+.pkl'%(base), f)]
    files.sort()
    
    for f in files:
        with open("%s/%s"%(dir,f), "rb") as fp:
            history = pickle.load(fp)
            plt.plot(history['loss'])
    plt.ylabel('MSE')
    plt.xlabel('epochs')
    plt.legend(files)
    
def create_parser():
    '''
    Create a command line parser for the experiment
    '''
    parser = argparse.ArgumentParser(prog='my Network')
    
    parser.add_argument('--exp', type=int, default=0, help='Experiment index')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    #parser.add_argument('--hidden', type=int, default=2, help='Number of Hidden Units')
    parser.add_argument('--hidden', nargs='+', type=int, default=[10, 10], help='Number of Hidden Units per layer (sequence of inputs)')
    parser.add_argument('--lrate', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbosity Level')
    
    return parser

'''
This next bit of code is executed only if this python file itself is executed
(if it is imported into another file, then the code below is not executed)
'''
if __name__ == "__main__":
    # Parse the command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Turn off GPU?
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # GPU check
    physical_devices = tf.config.list_physical_devices('GPU') 
    n_physical_devices = len(physical_devices)
    if(n_physical_devices > 0):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('We have %d GPUs\n'%n_physical_devices)
    else:
        print('NO GPU')

    # Do the work
    execute_exp(args)
