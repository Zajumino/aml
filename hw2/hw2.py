'''
Advanced Machine Learning, 2023
Homework 1 & 2

Building predictors for brain-machine interfaces

Author: Andrew H. Fagg
Modified by: Alan Lee
'''
# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import os
import sys
import time
import pickle
import argparse

# Location for libraries (you will likely just use './')
tf_tools = "./tf_tools/"

sys.path.append(tf_tools + "metrics")
sys.path.append(tf_tools + "networks")
sys.path.append(tf_tools + "experiment_control")

from deep_networks import *
from symbiotic_metrics import *
from job_control import *


#################################################################
def extract_data(bmi, args):
    '''
    Translate BMI data structure from the file into a data set for training/evaluating a single model
    
    :param bmi: Dictionary containing the full BMI data set, as loaded from the pickle file.
    :param args: Argparse object, which contains key information, including Nfolds, Ntraining,
            predict_dim, output_type, rotation
            
    :return: Numpy arrays in standard TF format for training set input/output, 
            validation set input/output and testing set input/output; and a
            dictionary containing the lists of folds that have been chosen
    '''
    # args to vars
    input_type = 'MI'
    output_type = args.output_type
    predict_dim = args.predict_dim
    Nfolds = args.Nfolds # number of folds in the data set
    Ntraining = args.Ntraining # number of folds to use for training
    r = args.rotation # rotation to use for training
    
    # Pull out input and output data
    ins = bmi[input_type]
    outs = bmi[output_type]
    
    times = bmi['time']
    
    # Check that argument matches actual number of folds
    assert (len(ins) == Nfolds), "Nfolds must match folds in data set"
    
    # Check that predict_dim is valid
    assert (predict_dim is None or (predict_dim >= 0 and predict_dim < outs[0].shape[1]))
    
    # Compute which folds belong in which set
    folds_training = (np.array(range(Ntraining)) + r) % Nfolds
    folds_validation = (np.array([Nfolds-2]) + r ) % Nfolds
    folds_testing = (np.array([Nfolds-1]) + r) % Nfolds
    
    # Log these choices
    folds = {'folds_training': folds_training, 'folds_validation': folds_validation, 'folds_testing': folds_testing}
    
    # Combine the folds into training/val/test data sets (pairs of input/output numpy arrays)
    ins_training = np.concatenate(np.take(ins,folds_training))
    outs_training = np.concatenate(np.take(outs,folds_training))
    time_training = np.concatenate(np.take(times,folds_training))
        
    ins_validation = np.concatenate(np.take(ins,folds_validation))
    outs_validation = np.concatenate(np.take(outs,folds_validation))
    time_validation = np.concatenate(np.take(times,folds_validation))
        
    ins_testing = np.concatenate(np.take(ins,folds_testing))
    outs_testing = np.concatenate(np.take(outs,folds_testing))
    time_testing = np.concatenate(np.take(times,folds_testing))
    
    # If a particular output dimension is specified, then extract it from the outputs
    if predict_dim is not None:
        outs_training = outs_training[:,[predict_dim]]
        outs_validation = outs_validation[:,[predict_dim]]
        outs_testing = outs_testing[:,[predict_dim]]
    
    return ins_training, outs_training, ins_validation, outs_validation, ins_testing, outs_testing, folds, time_training, time_validation, time_testing



def exp_type_to_hyperparameters(args):
    '''
    Translate the exp_type into a hyperparameter set

    :param args: ArgumentParser
    :return: Hyperparameter set (in dictionary form)
    '''
    
    if args.exp_type == 'noreg':
        p = {'Ntraining': [1,2,3,5,9,13,18], 
             'rotation': range(20)}
    elif args.exp_type == 'dropout':
        p = {'dropout': [0.1, 0.25, 0.4, 0.5, 0.7],
             'Ntraining': [1,2,3,5,9,13,18], 
             'rotation': range(20)}
    elif args.exp_type == 'lpreg':
        p = {'l1': [1, 0.1, 0.01, 0.001, 0.0001],
             'Ntraining': [1,2,3,5,9,13,18], 
             'rotation': range(20)}
    elif args.exp_type == 'bmi':
        p = {'Ntraining': [1,2,3,4,5,9,13,18], 
             'rotation': range(20)}
    else: 
        assert False, "Bad exp_type"

    return p

def augment_args(args):
    '''
    Use the jobiterator to override the command-line arguments based on the experiment index. 

    @return A string representing the selection of parameters to be used in the file name
    '''
    index = args.exp_index
    if(index is None):
        # UPDATE
        return "Ntraining_%d_rotation_%d"%(args.Ntraining, args.rotation)
    
    # Create parameter sets to execute the experiment on.
    # This defines the Cartesian product of experiments that we will be executing
    p = exp_type_to_hyperparameters(args)

    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())
    
    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index < ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))
    
    # Push the attributes to the args object and return a string that describes these structures
    return ji.set_attributes_by_index(args.exp_index, args)
    
def generate_fname(args, params_str):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.
    This way, they are unique and easy to identify after the fact.
    
    Expand this as needed
    '''
    # Hidden unit configuration
    hidden_str = '_'.join(str(x) for x in args.hidden)
    
    # Dimension being predicted
    if args.predict_dim is None:
        predict_str = args.output_type
    else:
        predict_str = '%s_%d'%(args.output_type, args.predict_dim)
        
    # Put it all together, including #of training folds and the experiment rotation
    return "%s/%s_%s_hidden_%s_%s"%(args.results_path, args.exp_type, predict_str, hidden_str, params_str)



def execute_exp(args=None):
    '''
    Perform the training and evaluation for a single model
    
    @args Argparse arguments
    '''
    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])
        
    # Modify the args if an exp_index is specified
    params_str = augment_args(args)
    
    print("Params:", params_str)
    
    # Compute output file name base
    fbase = generate_fname(args, params_str)
    
    print("File name base:", fbase)

    # Output pickle file name
    fname_out = "%s_results.pkl"%(fbase)

    # Check if this file exists
    if os.path.exists(fname_out):
        # File exists: abort the run
        print("File already exists")
        return None
    
    # Is this a test run?
    if(args.nogo):
        # Don't execute the experiment
        print("Test run only")
        return None
    
    # Set number of threads, if it is specified
    #  This limits the number of threads that you use on the machine (should not exceed what
    #  you have reserved on the supercomputer)
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)

    # Load the data
    fp = open(args.dataset, "rb")
    bmi = pickle.load(fp)
    fp.close()
    
    # Extract the data sets.  This process uses rotation and Ntraining (among other exp args)
    ins, outs, ins_validation, outs_validation, ins_testing, outs_testing, folds, time_training, time_validation, time_testing = extract_data(bmi, args)
    
    # Metrics
    fvaf = FractionOfVarianceAccountedFor(outs.shape[1])
    rmse = tf.keras.metrics.RootMeanSquaredError()

    # Build the model: you are responsible for providing this function
    model = deep_network_basic(ins.shape[1], tuple(args.hidden), outs.shape[1],
                               activation=args.activation_hidden, activation_out=args.activation_out,
                               lrate=args.lrate, metrics=[fvaf, rmse])
    
    # Report if verbosity is turned on
    if args.verbose >= 1:
        print(model.summary())
    
    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                      restore_best_weights=True,
                                                      min_delta=args.min_delta)

    # Could add callback for tensorboard...
    
    # Learn
    history = model.fit(x=ins, y=outs, epochs=args.epochs, verbose=args.verbose>=2,
                        validation_data=(ins_validation, outs_validation), 
                        callbacks=[early_stopping_cb])
        
    # Generate log data
    results = {}
    results['args'] = args
    
    results['predict_training'] = model.predict(ins)
    results['predict_training_eval'] = model.evaluate(ins, outs)
    results['outs_training'] = outs
    results['time_training'] = time_training
    
    results['predict_validation'] = model.predict(ins_validation)
    results['predict_validation_eval'] = model.evaluate(ins_validation, outs_validation)
    results['outs_validation'] = outs_validation
    results['time_validation'] = time_validation
    
    results['predict_testing'] = model.predict(ins_testing)
    results['predict_testing_eval'] = model.evaluate(ins_testing, outs_testing)
    results['outs_testing'] = outs_testing
    results['time_testing'] = time_testing
    
    results['folds'] = folds
    
    results['history'] = history.history
    
    # Save results
    results['fname_base'] = fbase
    fp = open(fname_out, "wb")
    pickle.dump(results, fp)
    fp.close()
    
    # Save the model (can't be included in the pickle file)
    model.save("%s_model"%(fbase))
    return model
               
def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='BMI Learner', fromfile_prefix_chars='@')

    # Problem definition
    parser.add_argument('--dataset', type=str, default='/home/fagg/datasets/bmi/bmi_dataset2.pkl', help='Data set file')
    parser.add_argument('--output_type', type=str, default='ddtheta', help='Type to predict')
    parser.add_argument('--predict_dim', type=int, default=None, help="Dimension of the output to predict")
    parser.add_argument('--Nfolds', type=int, default=20, help='Maximum number of folds')

    # Network details
    parser.add_argument('--activation_out', type=str, default=None, help='Activation for output layer')
    parser.add_argument('--activation_hidden', type=str, default='elu', help='Activation for hidden layers')
    parser.add_argument('--hidden', nargs='+', type=int, default=[400, 200, 100, 50, 25, 12], help='Number of hidden units per layer (sequence of ints)')

    # Experiment details
    parser.add_argument('--rotation', type=int, default=0, help='Cross-validation rotation')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--Ntraining', type=int, default=2, help='Number of training folds')

    # Meta experiment details
    parser.add_argument('--exp_type', type=str, default='bmi', help='High level name for this set of experiments; selects the specific Cartesian product')
    parser.add_argument('--exp_index', type=int, default=None, help='Experiment index for Cartesian experiment')

    # Training parameters
    parser.add_argument('--lrate', type=float, default=0.001, help="Learning rate")

    # Regularization
    parser.add_argument('--dropout', type=float, default=None, help="Dropout rate")
    parser.add_argument('--L1_regularization', '--l1', type=float, default=None, help="L1 regularization factor (only active if no L2)")
    parser.add_argument('--L2_regularization', '--l2', type=float, default=None, help="L2 regularization factor")

    # Early stopping
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")

    # Computer config
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--cpus_per_task', type=int, default=None, help='Number of threads to use')

    # Results
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")

    # Execution control
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    
    return parser

def check_args(args):
    '''
    Check that key arguments are within appropriate bounds.  Failing an assert causes a hard failure with meaningful output
    '''
    assert (args.rotation >= 0 and args.rotation < args.Nfolds), "Rotation must be between 0 and Nfolds"
    assert (args.Ntraining >= 1 and args.Ntraining <= (args.Nfolds-2)), "Ntraining must be between 1 and Nfolds-2"
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"


def check_completeness(args):
    '''
    Check the completeness of a Cartesian product run.

    All other args should be the same as if you executed your batch, however, the '--check' flag has been set

    Prints a report of the missing runs, including both the exp_index and the name of the missing results file

    :param args: ArgumentParser
    
    '''
    
    # Get the corresponding hyperparameters
    p = exp_type_to_hyperparameters(args)

    # Create the iterator
    ji = JobIterator(p)

    print("Total jobs: %d"%ji.get_njobs())

    print("MISSING RUNS:")

    indices = []
    # Iterate over all possible jobs
    for i in range(ji.get_njobs()):
        params_str = ji.set_attributes_by_index(i, args)
        # Compute output file name base
        fbase = generate_fname(args, params_str)
    
        # Output pickle file name
        fname_out = "%s_results.pkl"%(fbase)

        if not os.path.exists(fname_out):
            # Results file does not exist: report it
            print("%3d\t%s"%(i, fname_out))
            indices.append(i)

    # Give the list of indices that can be inserted into the --array line of the batch file
    print("Missing indices (%d): %s"%(len(indices),','.join(str(x) for x in indices)))

#################################################################
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)
    
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

    if(args.check):
        # Just look at which results files have NOT been created yet
        check_completeness(args)
    else:
        # Do the work
        execute_exp(args)
