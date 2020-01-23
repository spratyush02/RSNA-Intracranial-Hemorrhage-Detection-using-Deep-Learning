import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tqdm import tqdm
import argparse
import time
import os

import model
import utils
import extract_tf_records as data

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=32, help="Batch size to use during training.")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training.")
parser.add_argument("--pretrained_model", type=str, default='inception_resnet', help="Base model used for feature extraction")

parser.add_argument("--save_dir", type=str, default='./experiments', help='Directory where model and config files are saved')
parser.add_argument("--data_dir", type=str, default='/cluster/scratch/aabhinav/rsna_data/', help='Directory where all the data is being stored')

parser.add_argument('--loss', type=str, default='bin_loss', help='loss to be used for training')
parser.add_argument('--export', action='store_true', help='Save model checkpoint or not?')


ARGS = parser.parse_args()
EXPERIMENT_TIMESTAMP = str(int(time.time()))

EPOCHS = ARGS.num_epochs
BATCHSIZE = ARGS.batch_size


train_input_path = os.path.join(ARGS.data_dir, 'train','train-*.tfrecord')
val_input_path = os.path.join(ARGS.data_dir, 'validation', 'train-*.tfrecord')
SEED = 88

input_shape = (None, 224, 224, 3)
label_shape = (None, 6)

# Creating save directory where model checkpoints and
# tensorboard files will be stored
save_dir = utils.create_save_dir(ARGS, EXPERIMENT_TIMESTAMP, 'base_model')

# loading resnet-50 or inception resnet computation
# graph and adding loss and optimizer to the graph
if ARGS.pretrained_model == 'resnet_50':
    outputs, inputs, labels, is_train, _ = model.resnet_model(input_shape, label_shape)
if ARGS.pretrained_model == 'inception_resnet':
    outputs, inputs, labels, is_train, _ = model.inception_resnet_model(input_shape, label_shape)
loss = model.build_loss(labels, outputs, loss_name=ARGS.loss)

prediction = tf.cast(tf.greater(tf.sigmoid(outputs), 0.5), tf.float32)
correct_prediction = tf.equal(labels[:,-1], prediction[:,-1])

correct_images = tf.boolean_mask(inputs, correct_prediction)
wrong_images = tf.boolean_mask(inputs, tf.logical_not(correct_prediction))

train_loss = tf.summary.scalar('train_loss', loss)
validation_loss = tf.summary.scalar('val_loss', loss)
# val_correct_img = tf.summary.image('correct_imgs', correct_images, max_outputs=5)
# val_wrong_img = tf.summary.image('wrong_imgs', wrong_images, max_outputs=5)

train_op, gradient_norm = model.optimizer(loss, lr=ARGS.lr)

grad_norm = tf.summary.scalar('grad_norm', gradient_norm)

train_summary = tf.summary.merge([train_loss, grad_norm])
validation_summary = tf.summary.merge([validation_loss])
# validation_summary = tf.summary.merge([validation_loss, val_correct_img, val_wrong_img])


# Creating train and validation data iterators using
# tensorflow Dataset api
train_iter, train_data = data.extract_image(train_input_path, BATCHSIZE, SEED, mode='train')
val_iter, val_data = data.extract_image(val_input_path, BATCHSIZE, SEED, shuffle=False, mode='validation')
   
with tf.Session() as sess:
    # Creating tensorboard writer object
    train_writer = tf.summary.FileWriter(os.path.join(save_dir, '.log/train'), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(save_dir, '.log/val'))
    global_step_train = 0
    global_step_val = 0

    # Loading pretrained checkpoints
    if ARGS.pretrained_model == 'resnet_50':
        ckpt_name = 'resnet_v2_50.ckpt'
    if ARGS.pretrained_model == 'inception_resnet':
        ckpt_name = 'inception_resnet_v2_2016_08_30.ckpt'
    
    saver, pretrained_vars = utils.load_checkpoint(ckpt_name, sess, 'inception_resnet')
    unloaded_variables = [v for v in tf.global_variables() if v not in pretrained_vars]

    # Initializing new variables not present in checkpoint
    saver_unloaded = tf.train.Saver(unloaded_variables, max_to_keep=1, save_relative_paths=True)
    sess.run(tf.initialize_variables(unloaded_variables))
    
    data_pl = [inputs, labels, is_train]
    run_objects_train = [train_op, prediction, loss, train_summary]
    run_objects_val = [train_op, prediction, loss, validation_summary]

    for epoch in tqdm(range(EPOCHS)):
        train_f1, train_loss, train_n_batch, epoch_step_train, train_writer = model.run(sess, train_iter, train_data, data_pl, run_objects_train, train_writer, global_step_train, training=True)
        val_f1, val_loss, val_n_batch, epoch_step_val, val_writer = model.run(sess, val_iter, val_data, data_pl, run_objects_val, val_writer, global_step_val, training=False)

        print('Epoch {} Train F1_score: {:.3f}, Train loss: {:.3f}, Val F1_score: {:.3f}, Val loss: {:.3f}'.format(epoch+1, train_f1/train_n_batch, train_loss/train_n_batch, val_f1/val_n_batch, val_loss/val_n_batch))

        global_step_train = epoch_step_train
        global_step_val = epoch_step_val

        if ARGS.export:
            utils.save_model(sess, saver, save_dir, ARGS, SEED, 'model')
            utils.save_model(sess, saver_unloaded, save_dir, ARGS, SEED, 'model_unsaved')