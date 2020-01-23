import os
import argparse
import cv2
import json
import pickle
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
parser.add_argument("--base_data", type=str, default='train', help="supports two values: test and train")
parser.add_argument("--max_to_keep", type=int, default=60, help="max number of time states to keep")
parser.add_argument("--data_dir", type=str, default="/cluster/scratch/aabhinav/rsna_data/", help="Directory where all tfrecord files are saved")

ARGS = parser.parse_args()

model_dir = glob('./experiments/{}_*'.format(ARGS.model_id))[0]
config = json.load(open(os.path.join(model_dir, 'config.json'), 'r'))

print(model_dir)

BATCH_SIZE = config['batch_size']
SEED = config['seed']
base_model = config['pretrained_model']
base_data = ARGS.base_data
DATA_DIR = ARGS.data_dir

if base_data == 'train':
    input_path = os.path.join(DATA_DIR,'*','train-*.tfrecord')
    mode = 'validation'
else:
    input_path = os.path.join(DATA_DIR, 'test', 'test.tfrecord')
    mode='test'

# Creating data iterator based on tensorflow Dataset api
test_iter, test_data = data.extract_image(input_path, BATCH_SIZE, SEED, mode=mode)


input_shape = (None, 224, 224, 3)
label_shape = (None, 6)

# Loading base model whose embedding layer will be extracted
if base_model == 'resnet_50':
    _, x, _, is_train, features = model.resnet_model(input_shape, label_shape)
    feature_size = 2048
elif base_model == 'inception_resnet':
    _, x, _, is_train, features = model.inception_resnet_model(input_shape, label_shape)
    feature_size = 1536
else:
    AssertionError('Unknown base model present')


sess = tf.Session()

# Loading trained model checkpoint
saver = utils.load_checkpoint(os.path.join(model_dir,'model'), sess, 'model')
saver_unloaded = utils.load_checkpoint(os.path.join(model_dir,'model_unsaved'), sess, 'model_unsaved')

# Extracting feature dictionary for the provided dataset
data_pl = [x, is_train]
feature_dict = model.extract_features(sess, test_iter, test_data, data_pl, features)
print(f'size of feature dict : {len(feature_dict)}')
print('Feature_extraction Done')

# Saving feature dictionary into pickle file
with open(os.path.join(DATA_DIR, 'train_features.pickle'), 'wb') as handle:
    pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Loading metadata which will be later used to create 
# temporal sequences
if base_data == 'train':
    df = pd.read_csv(os.path.join(DATA_DIR, 'train_metadata.csv'), sep=',')
    df_labels = pd.read_csv(os.path.join(DATA_DIR, 'reshaped_stage_2_train.csv'), sep=',')

    df = pd.merge(df, df_labels, left_on='SOPInstanceUID', right_on='ImgID')
    df.sort_values(by=['PatientID', 'ImagePositionPatient'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    MAX_TO_KEEP = ARGS.max_to_keep
else:
    df = pd.read_csv(os.path.join(DATA_DIR, 'test_metadata.csv'), sep=',')
    MAX_TO_KEEP = df.groupby('PatientID').size().max()


embedding_arr = []
label_arr = []
mask_arr = []
img_ids_arr = []
count = 0
# counter = 0
disease_types = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
for patient_name, group in tqdm(df.groupby('PatientID')):
    feature_arr = []
    group_size = group.shape[0]
    if group_size>=MAX_TO_KEEP:
        # If number of slices for each patient is more than
        # max time cells in sequence model, then perform
        # uniform interval sampling to preserve temporal structure
        row_ids = [int(i) for i in np.linspace(0,group_size-1,MAX_TO_KEEP)]
        ids = list(group['SOPInstanceUID'].values[row_ids])
        if base_data == 'train':
            labels = group[disease_types].values[row_ids]
            labels = labels.astype(np.float32)
            mask = [1]*MAX_TO_KEEP

    else:
        # If number of slices less than time cells in sequence
        # model, then pad the start of sequence with random vector
        size_deficit = MAX_TO_KEEP - group_size
        ids = ['pad']*size_deficit
        ids += list(group['SOPInstanceUID'].values)

        if base_data == 'train':
            pad_labels = np.zeros((size_deficit,6), dtype=np.float32)
            labels = group[disease_types].values
            labels = labels.astype(np.float32)
            labels = np.concatenate([pad_labels,labels])

            # Creating mask to identify padded sequences
            mask = [0]*size_deficit
            mask += [1]*group_size

    for id in ids:
        try:
            feature_arr.append(feature_dict[id+'.jpg'])
        except:
            feature_arr.append(np.random.rand(feature_size))
            count += 1
            # continue
    embedding_arr.append(feature_arr)
    if base_data == 'train':
        label_arr.append(labels)
        mask_arr.append(mask)
    if base_data == 'test':
        img_ids_arr.append(ids)

embedding_arr = np.array(embedding_arr)
if base_data == 'train':
    label_arr = np.array(label_arr)
    mask_arr = np.array(mask_arr)
if base_data == 'test':
    img_ids_arr = np.array(img_ids_arr)


print(f'Null embeddings count: {count}')

# Saving all the numpy array in .npy file
if base_data == 'train':
    np.save(os.path.join(DATA_DIR,'train_embeddings.npy'),embedding_arr)
    np.save(os.path.join(DATA_DIR,'train_labels.npy'),label_arr)
    np.save(os.path.join(DATA_DIR,'train_mask.npy'),mask_arr)
else:
    np.save(os.path.join(DATA_DIR,'test_embeddings.npy'),embedding_arr)
    np.save(os.path.join(DATA_DIR,'test_img_ids.npy'),img_ids_arr)

sess.close()