import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tqdm import tqdm
import argparse
import time
import os
from glob import glob
import json
from sklearn.model_selection import train_test_split

import model
import utils
import extract_tf_records as data

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=32, help="Batch size to use during training.")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
parser.add_argument("--num_layers", type=int, default=2, help="Number of training epochs.")
parser.add_argument("--cell_size", type=int, default=256, help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training.")
parser.add_argument("--val_ratio", type=float, default=0.1, help='Validation split ratio')
parser.add_argument("--dropout_rate", type=float, default=0.7, help='Dropout rate for FC layers')

parser.add_argument("--save_dir", type=str, default='./experiments/lstm', help='Directory where model and config files are saved')
parser.add_argument("--data_dir", type=str, default='/cluster/scratch/aabhinav/rsna_data/', help='Directory where all the data is being stored')

parser.add_argument('--loss', type=str, default='masked_bin_loss', help='loss to be used for training')
parser.add_argument('--export', action='store_true', help='Save model checkpoint or not?')


ARGS = parser.parse_args()
EXPERIMENT_TIMESTAMP = str(int(time.time()))

EPOCHS = ARGS.num_epochs
BATCHSIZE = ARGS.batch_size
NUM_LAYERS = ARGS.num_layers
CELL_SIZE = ARGS.cell_size
VAL_RATIO = ARGS.val_ratio
DROP_RATE = ARGS.dropout_rate

# Creating folder where model checkpoints will be saved
save_dir = utils.create_save_dir(ARGS, EXPERIMENT_TIMESTAMP, 'lstm')

# loading training data
data_path = ARGS.data_dir
data_embeddings = np.load(os.path.join(data_path, 'train_embeddings.npy'))
data_labels = np.load(os.path.join(data_path, 'train_labels.npy'))
data_mask = np.load(os.path.join(data_path, 'train_mask.npy'))
data_mask = data_mask.astype(np.float32)

# Checking if data shape is consistent or not
assert data_embeddings.shape[0] == data_labels.shape[0] == data_mask.shape[0]
print(data_embeddings.dtype)
print(data_labels.dtype)
print(data_mask.dtype)

num_time = data_labels.shape[1]
num_labels = data_labels.shape[2]

feature_size = data_embeddings.shape[-1]
seed = 88

# Creating bi-lstm based computational graph and attaching
# loss and optimizer to the graph
outputs, inputs, labels, drop_rate = model.lstm_model(input_shape=(None, num_time, feature_size), label_shape=(None, num_time, num_labels),
                                          num_layers=NUM_LAYERS, cell_size=CELL_SIZE)

loss, mask = model.build_loss(labels, outputs, loss_name=ARGS.loss)

patient_pred = model.compute_patient_prediction(labels, outputs, mask)

train_loss = tf.summary.scalar('train_loss', loss)
validation_loss = tf.summary.scalar('val_loss', loss)

train_op, gradient_norm = model.optimizer(loss, lr=ARGS.lr)

grad_norm = tf.summary.scalar('grad_norm', gradient_norm)

train_summary = tf.summary.merge([train_loss, grad_norm])
validation_summary = tf.summary.merge([validation_loss])

saver = tf.train.Saver(tf.global_variables(), max_to_keep=1, save_relative_paths=True)


# Dividing training data into train and validation data
train_embedding, val_embedding, train_labels, val_labels, train_mask, val_mask = train_test_split(data_embeddings, data_labels,
                                                                                                  data_mask, test_size=VAL_RATIO, 
                                                                                                  random_state=seed)

train_data_num = train_embedding.shape[0]
valid_data_num = val_embedding.shape[0]
print(f'Valid_Data Size: {valid_data_num}')

# Preparing train and validation data into dictionary for
# shuffler and batcher function
train_data = {'inputs': train_embedding, 'labels': train_labels, 'mask': train_mask}
val_data = {'inputs': val_embedding, 'labels': val_labels, 'mask': val_mask}

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Creating tensorboard writer objects
train_writer = tf.summary.FileWriter(os.path.join(save_dir, '.log/train'), sess.graph)
val_writer = tf.summary.FileWriter(os.path.join(save_dir, '.log/val'))
global_step_train = 1
global_step_val = 1

for idx, epoch in tqdm(enumerate(range(EPOCHS))):

    n_batches_train = train_data_num//BATCHSIZE if train_data_num%BATCHSIZE == 0 else train_data_num//BATCHSIZE+1
    train_data_epoch = utils.shuffler_and_batcher(data_num=train_data_num, input_data=train_data, 
                                            n_splits=n_batches_train, shuffle=True, seed=seed)

    for i in range(n_batches_train):
        feed_dict = {inputs: train_data_epoch['inputs'][i], labels: train_data_epoch['labels'][i], 
                     mask: train_data_epoch['mask'][i], drop_rate:DROP_RATE}
        # feed_dict = {k: v[i] for k,v in train_data.items()}
        _, batch_train_loss, train_loss_summ = sess.run([train_op, loss, train_summary], feed_dict=feed_dict)
        train_writer.add_summary(train_loss_summ, global_step_train)

        global_step_train+=1

    n_batches_valid = valid_data_num//BATCHSIZE if valid_data_num%BATCHSIZE == 0 else valid_data_num//BATCHSIZE+1
    valid_data_epoch = utils.shuffler_and_batcher(data_num=valid_data_num, input_data=val_data, 
                                      n_splits=n_batches_valid, shuffle=False, seed=seed)

    # print(f'Number of Validation Batches: {n_batches_valid}')
    val_f1 = 0
    val_loss = 0
    f1_counter = 0
    for i in range(n_batches_valid):
        feed_dict = {inputs: valid_data_epoch['inputs'][i], labels: valid_data_epoch['labels'][i], 
                     mask: valid_data_epoch['mask'][i], drop_rate:1.0}
        
        batch_val_loss, val_loss_summ, patient_pred_val = sess.run([loss, validation_summary, patient_pred], feed_dict=feed_dict)
        val_writer.add_summary(val_loss_summ, global_step_val)
        val_loss += batch_val_loss/n_batches_valid

        patient_gt = np.sum(valid_data_epoch['labels'][i][:,:,-1],-1)>2
        patient_gt = patient_gt.astype(int)
        if np.sum(patient_gt)>0:
            val_f1 += f1_score(patient_gt, patient_pred_val)
            f1_counter += 1

        global_step_val+=1
    
    print(f'Validation loss: {val_loss}')
    if f1_counter>0:
        print(f'Validation F1 score {val_f1/f1_counter}')
    else:
        print('F1 score not valid')

    if ARGS.export:
        kwargs = {'feature_size': feature_size, 'num_time': num_time}
        utils.save_model(sess, saver, save_dir, ARGS, seed, 'lstm_model', **kwargs)

sess.close()