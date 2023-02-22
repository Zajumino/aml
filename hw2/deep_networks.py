'''
Author: I-Kang Kennedy
'''

import tensorflow as tf
from tensorflow import keras

def deep_network_basic(n_inputs, n_hidden, n_output, activation='relu', activation_out=None, dropout=None, l1=None, l2=None, lrate=0.001, metrics=None):
    '''
    Build a simple dense model with regularization options

    :param n_inputs: Number of input dimensions
    :param n_hidden: Number of hidden units per layer (sequence of ints)
    :param n_output: Number of ouptut dimensions
    :param activation: Activation function to be used for hidden units
    :param activation_out: Activation function to be used for output units
    :param dropout: Dropout rate (for input layer and all hidden layers)
    :param l1: l1 regularization hyperparameter
    :param l2: l2 regularization hyperparameter
    :param lrate: Learning rate for Adam Optimizer
    :param metrics: Metrics for evaluation
    '''
    
    regularizer = None
    if l1 is not None and l2 is not None:
        regularizer = tf.keras.regularizers.l1_l2(l1, l2)
    elif l1 is not None:
        regularizer = tf.keras.regularizers.l1(l1)
    elif l2 is not None:
        regularizer = tf.keras.regularizers.l2(l2)
    
    # Sequential model
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(tf.keras.layers.InputLayer(input_shape=(n_inputs,)))
    if dropout is not None:
        model.add(tf.keras.layers.Dropout(dropout))
    
    # Hidden layers
    for i, n in enumerate(n_hidden):
        model.add(tf.keras.layers.Dense(n, use_bias=True, name='hidden_{:d}'.format(i), kernel_regularizer=regularizer, activation=activation))
        if dropout is not None:
            model.add(tf.keras.layers.Dropout(dropout))
    
    # Output layer
    model.add(tf.keras.layers.Dense(n_output, use_bias=False, name='output', activation=activation_out))
    
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lrate)
    
    # Compile the model with MSE loss
    model.compile(optimizer=opt, loss='mse', metrics=metrics)
    
    return model