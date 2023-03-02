'''
Author: I-Kang Kennedy
'''

import tensorflow as tf
from keras import layers

def create_cnn_classifier_network(input_shape, in_channels, out_classes, filters, kernel_size, strides, padding='valid', activation='relu', downsampling_mode='stride', flatten=False, fc=[], dropout=None, spatial_dropout=None, lambda_l2=None, lrate=0.001, loss='sparse_categorical_crossentropy', metrics=None):

    '''Description

    :param input_shape: Shape of input image
    :param in_channels: Number of input channels
    :param out_classes: Number of output classes

    :param filters: Convolution filters per layer (sequence of ints)
    :param kernel_size: Convolution filter size per layer (sequence of ints)
    :param strides: Convolution stride distance or pooling size (sequence of ints)
    :param padding: Padding type for convolutional layers - 'valid|same|zero' (default: 'valid')
    :param downsampling_mode: Downsampling mode used in convolutional layers - 'stride|max' (default: 'stride')
    :param activation: Activation function for each conv/dense layer (default: 'relu')

    :param flatten: True = Flatten, False = GlobalMaxPool
    :param fc: Fully connected units per layer preceding softmax output (sequence of ints)

    :param batch_norm:
    :param dropout: Dropout rate
    :param spatial_dropout: Dropout rate for convolutional layers
    :param lambda_l2: l2 regularization hyperparameter
    :param lrate: Learning rate for Adam Optimizer
    :param loss: loss function
    :param metrics: Metrics for evaluation
    '''

    assert len(filters) == len(kernel_size) == len(strides)

    regularizer = None
    if lambda_l2 is not None:
        regularizer = tf.keras.regularizers.l2(lambda_l2)


    # Sequential model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    # Convolutional layers
    for i in range(len(filters)):
        if downsampling_mode == 'stride':
            model.add(tf.keras.layers.Conv2D(filters[i], kernel_size[i], strides[i], padding=padding, activation=activation, use_bias=True))
        else:
            model.add(tf.keras.layers.Conv2D(filters[i], kernel_size[i], padding=padding, activation=activation, use_bias=True))
        if spatial_dropout is not None:
            model.add(tf.keras.layers.SpatialDropout2D(spatial_dropout))
        if downsampling_mode == 'max' and strides[i] > 1:
            model.add(tf.keras.layers.MaxPooling2D(strides[i]))
    
    # Flatten option
    if flatten:
        model.add(tf.keras.layers.Flatten())
    else:
        model.add(tf.keras.layers.GlobalMaxPool2D())
    
    # Dense layers
    for units in fc:
        model.add(tf.keras.layers.Dense(units, activation=activation))
        if dropout is not None:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(out_classes, activation='softmax', use_bias=False))
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lrate)
    
    # Compile the model with MSE loss
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model