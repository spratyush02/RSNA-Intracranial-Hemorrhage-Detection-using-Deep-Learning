import os
import argparse
import pandas as pd
import numpy as np
import cv2
import math
import tensorflow as tf
from random import shuffle

from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default='/cluster/scratch/aabhinav/rsna_data', help="Directory path where all the data is being saved")

ARGS = parser.parse_args()

TRAIN_IMG_PATHS = os.path.join(ARGS.data_dir,'rsna-intracranial-hemorrhage-detection/stage_2_train/*.jpg')
TEST_IMG_PATHS = os.path.join(ARGS.data_dir,'rsna-intracranial-hemorrhage-detection/stage_2_test/*.jpg')
PROCESSED_LABELS_PATH = os.path.join(ARGS.data_dir,'processed_stage_2_train.csv')
SAVE_DIR = ARGS.data_dir

def create_label_dict(df):
    '''
    Returns dictionary with keys are image id and value as array of labels for all six types
    '''
    label_dict = {}
    groups = df.groupby('ImgID')
    for group in tqdm(groups):
        label_dict[group[0]] = group[1]['Label'].values
    return label_dict


def convert_image(img_path, label_dict):
    '''
    Return object of Example class which contains image filename, row count, column count, image array and correspoding labels
    '''
    img_id = img_path.split('/')[-1][:-4]
    label = label_dict[img_id]
    img_shape = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).shape    #pylint: disable=no-member
    filename = os.path.basename(img_path)

    # Read image data in terms of bytes
    with tf.gfile.FastGFile(img_path, 'rb') as fid:
        image_data = fid.read()

    example = tf.train.Example(features = tf.train.Features(feature = {
        'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [filename.encode('utf-8')])),
        'rows': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[0]])),
        'cols': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[1]])),
        'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_data])),
        'label': tf.train.Feature(int64_list = tf.train.Int64List(value = label)),
    }))
    return example


def convert_image_folder(label_dict, out_dir, img_paths, tfrecord_name_prefix, file_num):
    '''
    Creates tfrecord files for all the training images
    '''
    count = 0
    i = 0
    num_of_images = len(img_paths)
    samples_per_file = math.ceil(num_of_images/file_num)
    pbar = tqdm(total = num_of_images+1)
    while i<num_of_images:
        tfrecord_file_name = '{}-{:02d}-of-{:02d}.tfrecord'.format(tfrecord_name_prefix, count+1, file_num)
        if count+1<19:
            file_save_dir = os.path.join(out_dir,'train', tfrecord_file_name)
        else:
            file_save_dir = os.path.join(out_dir,'validation',tfrecord_file_name)
        file_sample_count = 0
        with tf.io.TFRecordWriter(file_save_dir) as writer:
            while i<num_of_images and file_sample_count<samples_per_file:
                example = convert_image(img_paths[i], label_dict)
                writer.write(example.SerializeToString())
                file_sample_count+=1
                i+=1
                pbar.update(1)
        count+=1
    pbar.close()


def convert_test_image(img_path):
    '''
    Return object of Example class which contains image filename, row count, column count, image array for test images
    '''
    img_shape = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).shape    #pylint: disable=no-member
    filename = os.path.basename(img_path)

    # Read image data in terms of bytes
    with tf.gfile.FastGFile(img_path, 'rb') as fid:
        image_data = fid.read()

    example = tf.train.Example(features = tf.train.Features(feature = {
        'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [filename.encode('utf-8')])),
        'rows': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[0]])),
        'cols': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[1]])),
        'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_data])),
    }))
    return example


def convert_test_image_folder(img_paths, out_dir, tfrecord_name_prefix):
    '''
    Creates tfrecord file for all the test images
    '''
    tfrecord_file_name = '{}.tfrecord'.format(tfrecord_name_prefix)
    file_save_dir = os.path.join(out_dir,'test',tfrecord_file_name)
    with tf.io.TFRecordWriter(file_save_dir) as writer:
        for path in img_paths:
            example = convert_test_image(path)
            writer.write(example.SerializeToString())

paths = glob(TEST_IMG_PATHS)
convert_test_image_folder(paths, SAVE_DIR, 'test')


labels_df = pd.read_csv(PROCESSED_LABELS_PATH)
label_dict = create_label_dict(labels_df)
paths = glob(TRAIN_IMG_PATHS)
shuffle(paths)
convert_image_folder(label_dict, SAVE_DIR, paths, 'train',20)