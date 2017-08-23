# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# %% Borrowed utils from here: https://github.com/pkmital/tensorflow_tutorials/
#import tensorflow as tf
import numpy as np
import csv

def conv2d(x, n_filters,
           k_h=5, k_w=5,
           stride_h=2, stride_w=2,
           stddev=0.02,
           activation=lambda x: x,
           bias=True,
           padding='SAME',
           name="Conv2D"):
    """2D Convolution with options for kernel size, stride, and init deviation.
    Parameters
    ----------
    x : Tensor
        Input tensor to convolve.
    n_filters : int
        Number of filters to apply.
    k_h : int, optional
        Kernel height.
    k_w : int, optional
        Kernel width.
    stride_h : int, optional
        Stride in rows.
    stride_w : int, optional
        Stride in cols.
    stddev : float, optional
        Initialization's standard deviation.
    activation : arguments, optional
        Function which applies a nonlinearity
    padding : str, optional
        'SAME' or 'VALID'
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Convolved input.
    """
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w', [k_h, k_w, x.get_shape()[-1], n_filters],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(
            x, w, strides=[1, stride_h, stride_w, 1], padding=padding)
        if bias:
            b = tf.get_variable(
                'b', [n_filters],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = conv + b
        return conv
    
def linear(x, n_units, scope=None, stddev=0.02,
           activation=lambda x: x):
    """Fully-connected network.
    Parameters
    ----------
    x : Tensor
        Input tensor to the network.
    n_units : int
        Number of units to connect to.
    scope : str, optional
        Variable scope to use.
    stddev : float, optional
        Initialization's standard deviation.
    activation : arguments, optional
        Function which applies a nonlinearity
    Returns
    -------
    x : Tensor
        Fully-connected output.
    """
    shape = x.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], n_units], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        return activation(tf.matmul(x, matrix))
    
# %%
def weight_variable(shape):
    '''Helper function to create a weight variable initialized with
    a normal distribution
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    #initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    initial = tf.zeros(shape)
    return tf.Variable(initial)

# %%
def bias_variable(shape):
    '''Helper function to create a bias variable initialized with
    a constant value.
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)

# %% 
def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    labels = np.array(labels).astype('int32')
    n_labels = labels.shape[0]
    index_offset = (np.arange(n_labels) * n_classes).astype('int32')
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot


def prepare_trainVal_img_list(img_list, num_subjs):
    #num_imgs_per_subj =np.zeros([num_subjs])
    id_label_list = []
    for row in img_list:
        id_label = int(row[8])
        #num_imgs_per_subj[id_label] += 1
        id_label_list.append(id_label)

    id_label_list = np.asarray(id_label_list)
    id_label_list = np.reshape(id_label_list, [-1])
    
    train_indices_list = []
    valid_indices_list= [] 
    eval_train_indices_list = []
    eval_valid_indices_list = [] 
    for i in range(num_subjs):
        print i
        curr_subj_idx = np.nonzero(id_label_list == i)[0]
        tmp = np.random.permutation(curr_subj_idx)
        per80 = np.floor(len(curr_subj_idx) * 0.8)
        t_inds = tmp[0:per80]
        v_inds = tmp[per80:]

        train_indices_list.append(t_inds)
        valid_indices_list.append(v_inds)

        eval_train_indices_list.append(t_inds[0])
        eval_valid_indices_list.append(v_inds[0])

    train_indices_list = np.asarray(train_indices_list)
    valid_indices_list = np.asarray(valid_indices_list)
    eval_train_indices_list = np.asarray(eval_train_indices_list)
    eval_valid_indices_list = np.asarray(eval_valid_indices_list)
    #print train_indices_list, train_indices_list.shape 
    
    train_indices_list = np.hstack(train_indices_list).astype('int')
    valid_indices_list = np.hstack(valid_indices_list).astype('int')
    eval_train_indices_list = np.hstack(eval_train_indices_list).astype('int')
    eval_valid_indices_list = np.hstack(eval_valid_indices_list).astype('int')
    print train_indices_list.shape, valid_indices_list.shape, eval_train_indices_list.shape, eval_valid_indices_list.shape
    
    img_list = np.asarray(img_list)
    print img_list.shape
    train_list = img_list[train_indices_list]
    valid_list = img_list[valid_indices_list]
    eval_train_list = img_list[eval_train_indices_list]
    eval_valid_list = img_list[eval_valid_indices_list]

    np.savez("Oxford_trainVal_data_3DSTN.npz", train_list=train_list, valid_list=valid_list, eval_train_list=eval_train_list, eval_valid_list=eval_valid_list)

        
def select_eval_img_list(img_list, num_subjs, save_file_name):
    # number of validation subjects

    id_label_list = []
    for row in img_list:
        id_label = int(row[8])
        id_label_list.append(id_label)

    id_label_list = np.asarray(id_label_list)
    id_label_list = np.reshape(id_label_list, [-1])


    eval_indices_list = []
    for i in range(num_subjs):
        print i
        curr_subj_idx = np.nonzero(id_label_list == i)[0]
        tmp = np.random.permutation(curr_subj_idx)
        inds = tmp[0:min(5, len(curr_subj_idx))]

        eval_indices_list.append(inds)


    eval_indices_list = np.asarray(eval_indices_list)
    eval_indices_list = np.hstack(eval_indices_list).astype('int')
    print eval_indices_list.shape

    img_list = np.asarray(img_list)
    print img_list.shape
    eval_list = img_list[eval_indices_list]

    np.savez(save_file_name, eval_list=eval_list)

    

    """
    # Record the number of images per subject                                                                                                                                                               
    num_imgs_per_subj =np.zeros([num_subjs])
    for row in valid_img_list:
        id_label = int(row[8])
        num_imgs_per_subj[id_label] += 1



    hist_subj = np.zeros([num_subjs])
    idx = 0
    count = 0
    for row in valid_img_list:
        count += 1
        print count
        image_key = row[0]                                               
        image_path = row[1]
        id_label = int(row[8])
         
        if idx >= num_subjs:
            break
        
        if hist_subj[idx] < min(1, num_imgs_per_subj[idx]):
            if id_label == idx:
                with open(save_file_name, "a") as f:                                                                                                                                         
                    f.write(image_key + "," + image_path + "," + row[2] + "," + row[3] + "," + row[4] + "," + row[5] + "," + row[6] + "," + row[7] + "," + str(id_label) + "\n")
                hist_subj[idx] += 1
        else:
            idx += 1
    """

def input_processing(images, pose_labels, id_labels, train_mean_vec, mean_labels, std_labels, num_imgs, image_size, num_classes):

    images = images.reshape([num_imgs, image_size, image_size, 3])
    pose_labels = pose_labels.reshape([num_imgs, 6])
    id_labels = id_labels.reshape([num_imgs, 1])

                                                                                                                                                               
    id_labels = dense_to_one_hot(id_labels, num_classes)

    # Subtract train image mean                                                                                                                                                  
    images = images / 255.
    train_mean_mat = train_mean_vec2mat(train_mean_vec, images)
    normalized_images = images - train_mean_mat
   
    # Normalize labels
    normalized_pose_labels = (pose_labels - mean_labels) / (std_labels + 0.000000000000000001)

                                
    return normalized_images, normalized_pose_labels, id_labels



def train_mean_vec2mat(train_mean, images_array):
        height = images_array.shape[1]
        width = images_array.shape[2]
        #batch = images_array.shape[0]                                                                                                                                                                       
        train_mean_R = np.matlib.repmat(train_mean[0],height,width)
        train_mean_G = np.matlib.repmat(train_mean[1],height,width)
        train_mean_B = np.matlib.repmat(train_mean[2],height,width)

        R = np.reshape(train_mean_R, (height,width,1))
        G = np.reshape(train_mean_G, (height,width,1))
        B = np.reshape(train_mean_B, (height,width,1))
        train_mean_image = np.append(R, G, axis=2)
        train_mean_image = np.append(train_mean_image, B, axis=2)

        return train_mean_image


def create_file_list(csv_file_path):

        with open(csv_file_path, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                csv_list = list(csvreader)

        return csv_list
