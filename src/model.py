import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.contrib.slim.nets import resnet_v2
import inception_resnet_v2
import tensorflow.contrib.slim as slim
import scipy
from tqdm import tqdm

import utils

def build_loss(labels, preds, loss_name):
    '''
    Computes different losses and return loss object for 
    computation graph
    '''
    if loss_name == 'l2':
        loss = tf.reduce_mean(tf.reduce_sum(tf.losses.mean_squared_error(labels, preds, reduction=tf.losses.Reduction.NONE),1))
    if loss_name == 'l1':
        loss = tf.reduce_mean(tf.reduce_sum(tf.losses.absolute_difference(labels, preds, reduction=tf.losses.Reduction.NONE),1))
    if loss_name == 'log_loss':
        loss = tf.reduce_mean(tf.reduce_sum(tf.losses.log_loss(labels, preds, reduction=tf.losses.Reduction.NONE),1))
    if loss_name == 'bin_loss':
        loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=preds),1))
    if loss_name == 'cat_loss':
        loss = tf.nn.softmax_cross_entropy_with_logits(labels, preds)
    if loss_name == 'weighted_loss':
        weights = tf.constant([1.0,1.0,1.0,1.0,1.0,2.0], dtype=tf.float32)
        loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=preds),weights),1))
    if loss_name == 'masked_bin_loss':
        num_time_states = labels.get_shape()[1]
        mask = tf.placeholder(shape=(None,num_time_states), dtype=tf.float32)
        loss = tf.divide(tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=preds),2),mask)),tf.reduce_sum(mask))

        return loss, mask

    return loss

def optimizer(loss, lr):
    '''
    Builds Adam optimizer and apply gradient clipping
    '''
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, 10)
        train_op = optimizer.apply_gradients(grads_and_vars=zip(clipped_gradients, params))
        

    return train_op, norm

def compute_accuracy(labels, preds):
    prediction = tf.sigmoid(preds)
    prediction = tf.cast(tf.greater(prediction, 0.5), tf.int32)
#     masked = tf.boolean(prediction, labels)
    accuracy = tf.reduce_sum(tf.multiply(prediction, labels))/tf.reduce_sum(labels)

    return accuracy

def compute_patient_prediction(labels, preds, mask):
    '''
    Return binary tensor of shape (batch_size,) where 1 means
    patient has hemorrhage and 0 means negative
    '''
    prediction = tf.sigmoid(preds)
    prediction = tf.cast(tf.greater(prediction, 0.5), tf.float32)
    
    patient_prediction = tf.cast(tf.greater(tf.reduce_sum(tf.multiply(prediction[:,:,-1],mask),-1),2),tf.int32)
    
    return patient_prediction

def resnet_model(input_shape, label_shape):
    '''
    Creates ResNet-50 model and return all placeholder and
    final logit layer
    '''
    inputs = tf.placeholder(shape=input_shape, dtype=tf.float32)
    labels = tf.placeholder(shape=label_shape, dtype=tf.float32)
    is_train = tf.placeholder(tf.bool)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, _ = resnet_v2.resnet_v2_50(inputs, is_training=is_train)
        print(net)
    out_squeeze = tf.reshape(tf.squeeze(net), [-1,2048])


    outputs = tf.layers.dense(out_squeeze, label_shape[1])

    return outputs, inputs, labels, is_train, out_squeeze

def inception_resnet_model(input_shape, label_shape):
    '''
    Creates Inception ResNet model and return all placeholder and
    final logit layer
    '''
    inputs = tf.placeholder(shape=input_shape, dtype=tf.float32)
    labels = tf.placeholder(shape=label_shape, dtype=tf.float32)
    is_train = tf.placeholder(tf.bool)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        net, _ = inception_resnet_v2.inception_resnet_v2(inputs, num_classes=None, is_training=False)
        print(net)
    out_squeeze = tf.reshape(tf.squeeze(net), [-1,1536])

    outputs = tf.layers.dense(out_squeeze, label_shape[1])

    return outputs, inputs, labels, is_train, out_squeeze

def lstm_model(input_shape, label_shape, num_layers, cell_size): 
    '''
    Creates Bi-LSTM model and returns all placeholder and final
    logit layer
    '''
    inputs = tf.placeholder(shape=input_shape, dtype=tf.float32)
    labels = tf.placeholder(shape=label_shape, dtype=tf.float32)
    drop_rate = tf.placeholder(dtype=tf.float32)
    
    cells_arr_fw = []
    cells_arr_bw = []
    for _ in range(num_layers):
        cell_fw = tf.nn.rnn_cell.LSTMCell(cell_size, initializer=tf.contrib.layers.xavier_initializer())
        cell_bw = tf.nn.rnn_cell.LSTMCell(cell_size, initializer=tf.contrib.layers.xavier_initializer())

        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=1, output_keep_prob=drop_rate)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=1, output_keep_prob=drop_rate)

        
        cells_arr_fw.append(cell_fw)
        cells_arr_bw.append(cell_bw)

    output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_arr_fw, cells_arr_bw, inputs, dtype=tf.float32)
    outputs = tf.layers.dense(output, label_shape[-1])

    return outputs, inputs, labels, drop_rate

def run(sess, iterator, input_data, data_pl, run_objects, writer, global_step, training):
    '''
    Runs training pipeline for tfrecord files and return loss, f1score, model prediction and tensorboard writer objects
    '''
    sess.run(iterator.initializer)
    epoch_f1 = 0
    epoch_loss = 0
    n_batch = 0

    try:
    #     Keep extracting data till TFRecord is exhausted
        while True:
            image_data = sess.run(input_data)
            feed_dict = {data_pl[0]:image_data['image'], data_pl[1]:image_data['label'], data_pl[2]:training}

            if training:
                _, preds, batch_loss, loss_summ = sess.run(run_objects, feed_dict=feed_dict)
            else:
                preds, batch_loss, loss_summ = sess.run(run_objects[1:], feed_dict=feed_dict)

            writer.add_summary(loss_summ, global_step)

            batch_f1 = f1_score(image_data['label'].flatten(), preds.flatten())
                        
            epoch_f1+=batch_f1
            epoch_loss+=batch_loss
            n_batch+=1
            global_step+=1

    except:
        pass

    return epoch_f1, epoch_loss, n_batch, global_step, writer

def extract_features(sess, iterator, input_data, data_pl, preds):
    '''
    Returns dictionary whose key is image_id and value is
    pre logit feature from trained base model
    '''
    sess.run(iterator.initializer)
    feature_dict = {}
    print('Extracting features')
    try:
        pbar = tqdm(total = 23400)
        while True:
            image_data = sess.run(input_data)
            batch_size = len(image_data['filename'])
            feed_dict = {data_pl[0]:image_data['image'], data_pl[1]:False}

            batch_features = sess.run(preds, feed_dict=feed_dict)

            for i in range(batch_size):
                filename = image_data['filename'][i].decode('ascii')
                feature_dict[filename] = batch_features[i]

            pbar.update(1)
            # break
        pbar.close()
    except:
        pass

    return feature_dict

def predict_v1(sess, iterator, input_data, data_pl, preds):
    '''
    Predicts disease probabilities for test data and
    returns a list of list containing predicted probabilites
    for base model
    '''
    sess.run(iterator.initializer)
    disease_types = ['_epidural', '_intraparenchymal', '_intraventricular', '_subarachnoid', '_subdural', '_any']
    prediction = []
    print('Starting prediction')
    try:
        # Keep extracting data till TFRecord is exhausted
        while True:
            image_data = sess.run(input_data)
            feed_dict = {data_pl[0]:image_data['image'], data_pl[1]:False}

            batch_prediction = sess.run(preds, feed_dict=feed_dict)
                        
            batch_prediction_probs = scipy.special.expit(batch_prediction)

            batch_prediction_probs_flatten = batch_prediction_probs.flatten()

            patient_ids = [id.decode("utf-8")[:-4] for id in image_data['filename']]
            # print(patient_ids)
            path_batch_repeat = np.repeat(patient_ids, 6)
            disease_types_tile = np.tile(disease_types, len(patient_ids))


            pred_tuples = list(zip(path_batch_repeat, disease_types_tile, batch_prediction_probs_flatten))
            prediction+=pred_tuples

    except:
        pass

    return prediction

def predict_lstm(sess, input_data, data_pl, preds, BATCHSIZE):
    '''
    Predicts disease probabilities for test data and
    returns a list of list containing predicted probabilites
    for sequence model
    '''
    disease_types = ['_epidural', '_intraparenchymal', '_intraventricular', '_subarachnoid', '_subdural', '_any']
    prediction = []
    print('Starting prediction')

    data_num = input_data['inputs'].shape[0]

    n_batches = data_num//BATCHSIZE if data_num%BATCHSIZE == 0 else data_num//BATCHSIZE+1
    input_data = utils.shuffler_and_batcher(data_num=data_num, input_data=input_data, 
                                            n_splits=n_batches, shuffle=False)

    for i in range(n_batches):
        feed_dict = {data_pl[0]: input_data['inputs'][i], data_pl[1]: 1}
    
        batch_prediction = sess.run(preds, feed_dict=feed_dict)
        batch_prediction_probs = scipy.special.expit(batch_prediction)
        batch_prediction_probs_flatten = batch_prediction_probs.flatten()

        img_ids = input_data['filename'][i].flatten()
        path_batch_repeat = np.repeat(img_ids, 6)

        batch_prediction_probs_flatten = batch_prediction_probs_flatten[path_batch_repeat!= 'pad']
        path_batch_repeat = path_batch_repeat[path_batch_repeat!='pad']
        img_ids = img_ids[img_ids!='pad']

        disease_types_tile = np.tile(disease_types, len(img_ids))
        
        pred_tuples = list(zip(path_batch_repeat, disease_types_tile, batch_prediction_probs_flatten))
        prediction+=pred_tuples

    return prediction



def predict(sess, x, is_train, batch_size, preds, x_input, img_paths):
    data_size = x_input.shape[0]
    n_batches = data_size//batch_size if data_size%batch_size==0 else data_size//batch_size+1
    x_split = np.array_split(x_input, n_batches)
    path_split = np.array_split(img_paths, n_batches)

    disease_types = ['_epidural', '_intraparenchymal', '_intraventricular', '_subarachnoid', '_subdural', '_any']

    prediction = []
    for x_batch, path_batch in zip(x_split, path_split):
        feed_dict = {x: x_batch, is_train: False}
        batch_prediction = sess.run(preds, feed_dict=feed_dict)
        batch_prediction_probs = scipy.special.expit(batch_prediction)

        batch_prediction_probs_flatten = batch_prediction_probs.flatten()

        patient_ids = [path.split('/')[-1][:-4] for path in path_batch]
        path_batch_repeat = np.repeat(patient_ids, 6)
        disease_types_tile = np.tile(disease_types, len(path_batch))

        pred_tuples = list(zip(path_batch_repeat, disease_types_tile, batch_prediction_probs_flatten))
        prediction+=pred_tuples

    return prediction