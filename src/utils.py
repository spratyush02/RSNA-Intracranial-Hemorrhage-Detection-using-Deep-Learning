import numpy as np
import cv2
import os
import json
import tensorflow as tf

def create_save_dir(args, timestamp, model_name):
    '''
    Creates new directory naming based on model
    hyperparameters and return path of created 
    directory
    '''
    if model_name == 'base_model':
        experiment_name_format = '{}_tf_{}_b{}_e{}'
        experiment_name = experiment_name_format.format(timestamp, args.loss, args.batch_size, args.num_epochs)
    if model_name == 'lstm':
        experiment_name_format = '{}_b{}_e{}_l{}_cs{}'
        experiment_name = experiment_name_format.format(timestamp, args.batch_size, 
                                                        args.num_epochs, args.num_layers, args.cell_size)

    save_dir = os.path.join(args.save_dir, experiment_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    return save_dir

def save_model(session, saver, experiment_dir, args, seed, ckpt_name, **kwargs):

    '''
    Save the trained model checkpoints and json file
    containing all the arguments passed while running the
    training python file
    '''
    
    print('===== Saving Model =====')
        
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    if len(kwargs)>0:
        for k,v in kwargs.items():
            config[k] = v
    config['seed'] = seed

    json.dump(config, open(os.path.join(experiment_dir, 'config.json'), 'w'), indent=4, sort_keys=True)
    saver.save(session, os.path.normpath(os.path.join(experiment_dir, ckpt_name)))

def load_checkpoint(ckpt_name, sess, model_name):
    """Restore the latest checkpoint found in `experiment_dir`."""

    ckpt_reader = tf.train.NewCheckpointReader(ckpt_name)
    variables_names = list(ckpt_reader.get_variable_to_shape_map().keys())

    variables = [v for v in tf.global_variables() if v.op.name in variables_names]
    saver = tf.train.Saver(variables, max_to_keep=1, save_relative_paths=True)

    print("Loading {} model checkpoint".format(model_name))
    saver.restore(sess, ckpt_name)

    return saver, variables


def load_latest_checkpoint(model_dir, sess, modelname, latest_checkpoint_name):
    """Restore the latest checkpoint found in `experiment_dir`."""

    ckpt_reader = tf.train.NewCheckpointReader(os.path.join(model_dir,modelname))
    variables_names = list(ckpt_reader.get_variable_to_shape_map().keys())

    variables = [v for v in tf.global_variables() if v.op.name in variables_names]
    saver = tf.train.Saver(variables, max_to_keep=1, save_relative_paths=True)

    ckpt = tf.train.get_checkpoint_state(model_dir, latest_filename=latest_checkpoint_name)

    if ckpt and ckpt.model_checkpoint_path:
        # Check if the specific checkpoint exists
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("Loading model checkpoint {}".format(modelname))
        saver.restore(sess, ckpt.model_checkpoint_path)

        return saver
    else:
    	raise ValueError("could not load checkpoint")

def shuffler_and_batcher(data_num, input_data, n_splits, shuffle=False, seed = 92):
    '''
    Takes input as dictionary of input data and returns
    another dictionary where data is shuffled if shuffle flag
    is True and divided into batches based on n_split value
    '''

    if shuffle:
        indices = np.arange(data_num)
        np.random.seed(seed)
        np.random.shuffle(indices)
        input_data = {k: v[indices] for k,v in input_data.items()}

    batched_data = {k: np.array_split(v, n_splits, axis=0) for k,v in input_data.items()}

    return batched_data