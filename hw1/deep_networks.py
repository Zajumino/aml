'''
Author: I-Kang Kennedy
'''

import tensorflow as tf
from tensorflow import keras

def deep_network_basic(n_inputs, n_hidden, n_output, activation='relu', activation_out=None, lrate=0.001, metrics=None):
    '''
    Build a simple dense model
    
    :param n_inputs: Number of input dimensions
    :param n_hidden: Number of hidden units per layer (sequence of ints)
    :param n_output: Number of ouptut dimensions
    :param activation: Activation function to be used for hidden units
    :param activation_out: Activation function to be used for output units
    :param lrate: Learning rate for Adam Optimizer
    :param metrics:
    '''
    # Sequential model
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(tf.keras.layers.InputLayer(input_shape=(n_inputs,)))
    
    # Hidden layers
    for i, n in enumerate(n_hidden):
        model.add(tf.keras.layers.Dense(n, use_bias=True, name='hidden_{:d}'.format(i), activation=activation))
    
    # Output layer
    model.add(tf.keras.layers.Dense(n_output, use_bias=False, name='output', activation=activation_out))
    
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lrate)
    
    # Compile the model with MSE loss
    model.compile(optimizer=opt, loss='mse', metrics=metrics)
    
    return model