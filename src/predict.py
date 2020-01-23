import os
import argparse
import cv2
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import pandas as pd

import utils
import model
import extract_tf_records as data

parser = argparse.ArgumentParser()

parser.add_argument("--model_id", type=int, help="model_id (timestamp) of saved model.")
parser.add_argument("--model_type", type=str, default='lstm', help='Possible options lstm and base')
parser.add_argument("--data_dir", type=str, default='/cluster/scratch/aabhinav/rsna_data/', help='Directory where all the data is being stored')
ARGS = parser.parse_args()

if ARGS.model_type == 'base':
    model_dir = glob('./experiments/{}_*'.format(ARGS.model_id))[0]
elif ARGS.model_type == 'lstm':
    model_dir = glob('./experiments/lstm/{}_*'.format(ARGS.model_id))[0]
else:
    AssertionError('Unknown Model Type mentioned')
config = json.load(open(os.path.join(model_dir, 'config.json'), 'r'))

print(model_dir)

BATCH_SIZE = config['batch_size']
SEED = config['seed']
DATA_DIR = ARGS.data_dir

if ARGS.model_type == 'base':
    base_model = config['pretrained_model']

    input_path = os.path.join(DATA_DIR,'test','test.tfrecord')
    test_iter, test_data = data.extract_image(input_path, BATCH_SIZE, SEED, mode='test')


    input_shape = (None, 224, 224, 3)
    label_shape = (None, 6)

    if base_model == 'resnet_50':
        preds, x, y, is_train, _ = model.resnet_model(input_shape, label_shape)
    elif base_model == 'inception_resnet':
        preds, x, y, is_train, _ = model.inception_resnet_model(input_shape, label_shape)
    else:
        AssertionError('Unknown base model present')

    sess = tf.Session()

    saver = utils.load_checkpoint(os.path.join(model_dir,'model'), sess, 'model')
    saver_unloaded = utils.load_checkpoint(os.path.join(model_dir,'model_unsaved'), sess, 'model_unsaved')

    data_pl = [x, is_train]

    test_pred = model.predict_v1(sess, test_iter, test_data, data_pl, preds)



if ARGS.model_type == 'lstm':
    num_layers = config['num_layers']
    cell_size = config['cell_size']
    feature_size = config['feature_size']
    num_time = config['num_time']
    dropout_rate = config['dropout_rate']


    outputs, inputs, _, drop_rate = model.lstm_model(input_shape=(None, 60, feature_size), label_shape=(None, 60, 6),
                                               num_layers=num_layers, cell_size=cell_size)
    
    data_path = DATA_DIR
    test_embeddings = np.load(os.path.join(data_path, 'test_embeddings.npy'))
    test_filenames = np.load(os.path.join(data_path, 'test_img_ids.npy'))
    test_data = {'inputs': test_embeddings, 'filename': test_filenames}

    assert test_embeddings.shape[0] == test_filenames.shape[0]

    sess = tf.Session()
    saver = utils.load_checkpoint(os.path.join(model_dir,'lstm_model'), sess, 'lstm_model')

    data_pl = [inputs, drop_rate]
    test_pred = model.predict_lstm(sess, test_data, data_pl, outputs, BATCH_SIZE)


test_df = pd.DataFrame.from_records(test_pred, columns=['PatientID', 'Type', 'Label'])
test_df['ID'] = test_df['PatientID'] + test_df['Type']

df_final = test_df[['ID', 'Label']]
df_final.to_csv('output2.csv', index=False)

sess.close()