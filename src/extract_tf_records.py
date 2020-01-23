import tensorflow as tf
import shutil
import os
import cv2
import numpy as np
from albumentations import (Transpose, ShiftScaleRotate, HorizontalFlip, Compose)

def extract_fn(tfrecord):
    # Extract features using the keys set during creation
    features = {
        'filename': tf.FixedLenFeature([], tf.string),
        'rows': tf.FixedLenFeature([], tf.int64),
        'cols': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([6], tf.int64)
    }

    # Extract the data record
    sample = tf.parse_single_example(tfrecord, features)

    sample['image'] = tf.image.decode_image(sample['image'])

    return sample

def extract_test_fn(tfrecord):
    # Extract features using the keys set during creation
    features = {
        'filename': tf.FixedLenFeature([], tf.string),
        'rows': tf.FixedLenFeature([], tf.int64),
        'cols': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
    }

    # Extract the data record
    sample = tf.parse_single_example(tfrecord, features)

    sample['image'] = tf.image.decode_image(sample['image'])
    return sample



def extract_image(data_path, batch_size, seed, num_parallel_calls=16, shuffle=True, mode='train'):

    # Pipeline of dataset and iterator 
    if mode=='test':
        dataset = tf.data.TFRecordDataset([data_path])
        dataset = dataset.shuffle(buffer_size=batch_size*10)
        dataset = dataset.map(extract_test_fn, num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(preprocess_test_func, num_parallel_calls=num_parallel_calls)
    else:
        dataset = tf.data.TFRecordDataset.list_files(data_path, seed=seed, shuffle=shuffle)
        dataset = dataset.apply(
                tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset,
                                                        cycle_length=num_parallel_calls, block_length=1,
                                                        sloppy=shuffle))
        dataset = dataset.shuffle(buffer_size=batch_size*10)
        dataset = dataset.map(extract_fn, num_parallel_calls=num_parallel_calls)
        if mode=='train':
            dataset = dataset.map(preprocess_func, num_parallel_calls=num_parallel_calls)
        if mode=='validation':
            dataset = dataset.map(preprocess_test_func, num_parallel_calls=num_parallel_calls)

    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_image_data = iterator.get_next()

    return iterator, next_image_data

def preprocess_test_func(tf_record):
    tf_record['image'] = 2*(tf_record['image']/255)-1

    return tf_record

def preprocess_func(tf_record):
    '''
    Returns preprocessed data using various data augmentation
    methods and finally normalizing pixel values between
    -1 and 1
    '''

    def _my_np_func(p):
        transform_train = Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                        rotate_limit=20, p=0.3, border_mode = cv2.BORDER_REPLICATE),
        Transpose(p=0.5)
        ])

        img = transform_train(image=p)
        # print(f'Transformed image: {img}')

        return img['image']

    processed = tf.py_func(_my_np_func, [tf_record['image']], tf.uint8)
    print(processed)
    tf_record['image'] = processed

    tf_record['image'] = 2*(tf_record['image']/255)-1

    return tf_record