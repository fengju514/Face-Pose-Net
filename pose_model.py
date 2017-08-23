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

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

#import sys
#sys.path.append('/staging/pn/fengjuch/transformer')
#from spatial_transformer import transformer
#from tf_utils import weight_variable, bias_variable, dense_to_one_hot

"""
HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')
"""

class ThreeD_Pose_Estimation(object):
  """ResNet model."""

  def __init__(self, images, labels, mode, ifdropout, keep_rate_fc6, keep_rate_fc7, lr_rate_fac, net_data, batch_size, mean_labels, std_labels):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    #self.hps = hps
    self.batch_size = batch_size
    self._images = images
    self.labels = labels
    
    self.mode = mode
    self.ifdropout = ifdropout
    self.keep_rate_fc6 = keep_rate_fc6
    self.keep_rate_fc7 = keep_rate_fc7
    self.ifadd_weight_decay = 0 #ifadd_weight_decay
    self.net_data = net_data
    self.lr_rate_fac = lr_rate_fac
    self._extra_train_ops = []
    self.optimizer = 'Adam'
    self.mean_labels = mean_labels
    self.std_labels = std_labels
    #self.train_mean_vec = train_mean_vec

  def _build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._build_model()
    
    if self.mode == 'train':
      self._build_train_op()
    
    #self.summaries = tf.merge_all_summaries()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    #with tf.variable_scope('init'):
     # x = self._images
     # print x, x.get_shape()
     # x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))
     # print x, x.get_shape()
    with tf.variable_scope('Spatial_Transformer'):
      x = self._images
      x = tf.image.resize_bilinear(x, tf.constant([227,227], dtype=tf.int32)) # the image should be 227 x 227 x 3
      print x.get_shape()
      self.resized_img = x
      theta = self._ST('ST2', x, 3, (16,16), 3, 16, self._stride_arr(1))
      #print "*** ", x.get_shape()
   

    #with tf.variable_scope('logit'):
    #  logits = self._fully_connected(theta, self.hps.num_classes)
    #  self.predictions = tf.nn.softmax(logits)
      #print "*** ", logits, self.predictions

    with tf.variable_scope('costs'):
      self.predictions = theta
      self.preds_unNormalized = theta * (self.std_labels + 0.000000000000000001) + self.mean_labels
      pred_dim1 = theta.get_shape()[0]
      pred_dim2 = theta.get_shape()[1]

      del theta
      #diff = self.predictions - self.labels
      #print diff
      
      #xent = tf.mul(diff, diff) #tf.nn.l2_loss(diff)
      #print xent
      #xent = tf.reduce_sum(xent, 1)
      pow_res = tf.pow(self.predictions-self.labels, 2)
      """
      print pow_res, pow_res.get_shape()
      const1 = tf.constant(1.0,shape=[pred_dim1, 3],dtype=tf.float32)
      const2 = tf.constant(1.0,shape=[pred_dim1, 3],dtype=tf.float32)
      #print const1, const2, const1.get_shape(), const2.get_shape()
      const = tf.concat(1,[const1, const2])
      print const, const.get_shape()
      cpow_res = tf.mul(const,pow_res) 
      xent = tf.reduce_sum(cpow_res,1)
      print xent
      """
      xent = tf.reduce_sum(pow_res,1)
      self.cost = tf.reduce_mean(xent, name='xent')
      #print self.cost
      
      #self.cost = tf.nn.l2_loss(diff)
      #  Add weight decay of needed
      if self.ifadd_weight_decay == 1:
        self.cost += self._decay()
      

      #self.train_step = tf.train.GradientDescentOptimizer(self.hps.lrn_rate).minimize(self.cost)

      #tf.scalar_summary('cost', self.cost)



  def conv(self, input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        #input_groups = tf.split(3, group, input)
        #kernel_groups = tf.split(3, group, kernel)
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        #conv = tf.concat(3, output_groups)
        conv = tf.concat(output_groups, 3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])





  def _ST(self, name, x, channel_x, out_size, filter_size, out_filters, strides):
    """ Spatial Transformer. """

    with tf.variable_scope(name):

      # zero-mean input [B,G,R]: [93.5940, 104.7624, 129.1863] --> provided by vgg-face
      """
      with tf.name_scope('preprocess') as scope:
        mean = tf.constant(tf.reshape(self.train_mean_vec*255.0, [3]), dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        x = x - mean
      """

      # conv1
      with tf.name_scope('conv1') as scope:
        #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
        conv1W = tf.Variable(self.net_data["conv1"]["weights"], trainable=True, name='W')
        conv1b = tf.Variable(self.net_data["conv1"]["biases"], trainable=True, name='baises')
        conv1_in = self.conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in, name='conv1')
        print x.get_shape(), conv1.get_shape()
        

        #maxpool1
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1 = tf.nn.max_pool(conv1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool1')
        print maxpool1.get_shape()
        

        #lrn1                                                                                                                   
        #lrn(2, 2e-05, 0.75, name='norm1')                                                                                      
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(maxpool1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias, name='norm1')



      # conv2
      with tf.name_scope('conv2') as scope:
        #conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv2W = tf.Variable(self.net_data["conv2"]["weights"], trainable=True, name='W')
        conv2b = tf.Variable(self.net_data["conv2"]["biases"], trainable=True, name='baises')
        conv2_in = self.conv(lrn1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in, name='conv2')
        print conv2.get_shape()



        #maxpool2                                                                                                              
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                                    
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool2 = tf.nn.max_pool(conv2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool2')
        print maxpool2.get_shape()



        #lrn2
        #lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(maxpool2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias, name='norm2')

        

      # conv3                                                                                                                                   
      with tf.name_scope('conv3') as scope:
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        conv3W = tf.Variable(self.net_data["conv3"]["weights"], trainable=True, name='W')
        conv3b = tf.Variable(self.net_data["conv3"]["biases"], trainable=True, name='baises')
        conv3_in = self.conv(lrn2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in, name='conv3')
        print conv3.get_shape()
    
      # conv4                                                                                                                                                            
      with tf.name_scope('conv4') as scope:
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        conv4W = tf.Variable(self.net_data["conv4"]["weights"], trainable=True, name='W')
        conv4b = tf.Variable(self.net_data["conv4"]["biases"], trainable=True, name='baises')
        conv4_in = self.conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in, name='conv4')
        print conv4.get_shape()

      # conv5                                                                                                                                             
      with tf.name_scope('conv5') as scope:
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5W = tf.Variable(self.net_data["conv5"]["weights"], trainable=True, name='W')
        conv5b = tf.Variable(self.net_data["conv5"]["biases"], trainable=True, name='baises')
        self.conv5b = conv5b
        conv5_in = self.conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in, name='conv5')
        print conv5.get_shape()

        #maxpool5
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool5')
        print maxpool5.get_shape(), maxpool5.get_shape()[1:], int(np.prod(maxpool5.get_shape()[1:]))
        
      
      # fc6
      with tf.variable_scope('fc6') as scope:
        #fc(4096, name='fc6')
        fc6W = tf.Variable(self.net_data["fc6"]["weights"], trainable=True, name='W')
        fc6b = tf.Variable(self.net_data["fc6"]["biases"], trainable=True, name='baises')
        self.fc6W = fc6W
        self.fc6b = fc6b
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b, name='fc6')
        print fc6.get_shape()
        if self.ifdropout == 1:
          fc6 = tf.nn.dropout(fc6, self.keep_rate_fc6, name='fc6_dropout')
            
      # fc7 
      with tf.variable_scope('fc7') as scope:
        #fc(4096, name='fc7')
        fc7W = tf.Variable(self.net_data["fc7"]["weights"], trainable=True, name='W')
        fc7b = tf.Variable(self.net_data["fc7"]["biases"], trainable=True, name='baises')
        self.fc7b = fc7b
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b, name='fc7')
        print fc7.get_shape()
        if self.ifdropout == 1:
          fc7 = tf.nn.dropout(fc7, self.keep_rate_fc7, name='fc7_dropout')
                                                                                                   
      # fc8  
      with tf.variable_scope('fc8') as scope:
        """
        #fc(6, relu=False, name='fc8')
        fc8W = tf.Variable(net_data["fc8"][0])
        fc8b = tf.Variable(net_data["fc8"][1])
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
        """

        # Move everything into depth so we can perform a single matrix multiplication.                            
        fc7 = tf.reshape(fc7, [self.batch_size, -1])
        dim = fc7.get_shape()[1].value
        #print "fc7 dim:\n"
        #print fc7.get_shape(), dim
        fc8W = tf.Variable(tf.random_normal([dim, 6], mean=0.0, stddev=0.01), trainable=True, name='W')                                                                    
        fc8b = tf.Variable(tf.zeros([6]), trainable=True, name='baises')                                                                                                      
        self.fc8b = fc8b
        theta = tf.nn.xw_plus_b(fc7, fc8W, fc8b)  

        """
        weights = self._variable_with_weight_decay('weights', shape=[dim, 6],
                                          stddev=0.04, wd=None) #wd=0.004)
        biases = self._variable_on_cpu('biases', [6], tf.constant_initializer(0.1))
        theta = tf.matmul(reshape, weights) + biases
        
        print theta.get_shape()
        """

        self.theta = theta
        self.fc8W = fc8W
        self.fc8b = fc8b
        # %% We'll create a spatial transformer module to identify discriminative
        # %% patches
        #h_trans = self._transform(theta, x, out_size, channel_x)
        #print h_trans.get_shape()
      return theta



  def _variable_with_weight_decay(self, name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.                                                                                                               
    Note that the Variable is initialized with a truncated normal distribution.                                                                                                                             
    A weight decay is added only if one is specified.                                                                                                                                                       
    Args:                                                                                                                                                                                                   
    name: name of the variable                                                                                                                                                                              
    shape: list of ints                                                                                                                                                                                     
    stddev: standard deviation of a truncated Gaussian                                                                                                                                                      
    wd: add L2Loss weight decay multiplied by this float. If None, weight                                                                                                                                   
        decay is not added for this Variable.                                                                                                                                                               
    Returns:                                                                                                                                                                                                
    Variable Tensor                                                                                                                                                                                         
    """
    dtype = tf.float32 #if FLAGS.use_fp16 else tf.float32                                                                                                                                                    
    var = self._variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
      weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)
    return var



  def _variable_on_cpu(self, name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.                                                                                                                                                    
    Args:                                                                                                                                                                                                       name: name of the variable                                                                                                                                                                              
    shape: list of ints                                                                                                                                                                                     
    initializer: initializer for Variable                                                                                                                                                                   
    Returns:                                                                                                                                                                                                
    Variable Tensor                                                                                                                                                                                         
    """
    with tf.device('/cpu:0'):
      dtype = tf.float32 # if FLAGS.use_fp16 else tf.float32                                                                                                                                                 
      var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var





  def _build_train_op(self):
    """Build training specific ops for the graph."""
    #self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    #tf.scalar_summary('learning rate', self.lrn_rate)
    """
    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)
    """
    if self.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.optimizer == 'Adam':
      optimizer = tf.train.AdamOptimizer(0.001 * self.lr_rate_fac)
    elif self.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
    
    """
    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)
    """

    self.train_op = optimizer.minimize(self.cost)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]
      #print x.get_shape(), params_shape
      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        tf.histogram_summary(mean.op.name, mean)
        tf.histogram_summary(variance.op.name, variance)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.info('image after unit %s', x.get_shape())
    return x

  def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                           activate_before_residual=False):
    """Bottleneck resisual unit with 3 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 1, in_filter, out_filter/4, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

    with tf.variable_scope('sub3'):
      x = self._batch_norm('bn3', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
      x += orig_x

    tf.logging.info('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        aaa = tf.nn.l2_loss(var)
        #print aaa
        # tf.histogram_summary(var.op.name, var)

    return tf.mul(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    #print "*** ", x.get_shape()
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    #print "*** ", w.get_shape()
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    #print "*** ", b.get_shape()
    aaa = tf.nn.xw_plus_b(x, w, b)
    #print "*** ", aaa.get_shape()
    return tf.nn.xw_plus_b(x, w, b)

 
  def _fully_connected_ST(self, x, out_dim):
    """FullyConnected layer for final output of the localization network in the spatial transformer"""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW2', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    initial = np.array([[1., 0, 0], [0, 1., 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()
    b = tf.get_variable('biases2', [out_dim],
                        initializer=tf.constant_initializer(initial))
    return tf.nn.xw_plus_b(x, w, b)

   

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])


  def _repeat(self, x, n_repeats):
    with tf.variable_scope('_repeat'):
      rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.pack([n_repeats, ])), 1), [1, 0])
      rep = tf.cast(rep, 'int32')
      x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
      return tf.reshape(x, [-1])


  def _interpolate(self, im, x, y, out_size, channel_x):
    with tf.variable_scope('_interpolate2'):
      # constants
      num_batch = self.hps.batch_size #tf.shape(im)[0]
      print num_batch
      height = tf.shape(im)[1]
      width = tf.shape(im)[2]
      channels = tf.shape(im)[3]
      print channels
      #channels = tf.cast(channels, tf.int32)
      #print channels
      x = tf.cast(x, 'float32')
      y = tf.cast(y, 'float32')
      height_f = tf.cast(height, 'float32')
      width_f = tf.cast(width, 'float32')
      out_height = out_size[0]
      out_width = out_size[1]
      zero = tf.zeros([], dtype='int32')
      #max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
      #max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
      
      max_y = tf.cast(height - 1, 'int32')
      max_x = tf.cast(width - 1, 'int32')
      # scale indices from [-1, 1] to [0, width/height]
      x = (x + 1.0)*(width_f) / 2.0
      y = (y + 1.0)*(height_f) / 2.0

      # do sampling
      x0 = tf.cast(tf.floor(x), 'int32')
      x1 = x0 + 1
      y0 = tf.cast(tf.floor(y), 'int32')
      y1 = y0 + 1

      x0 = tf.clip_by_value(x0, zero, max_x)
      x1 = tf.clip_by_value(x1, zero, max_x)
      y0 = tf.clip_by_value(y0, zero, max_y)
      y1 = tf.clip_by_value(y1, zero, max_y)
      dim2 = width
      dim1 = width*height
      base = self._repeat(tf.range(num_batch)*dim1, out_height*out_width)
      base_y0 = base + y0*dim2
      base_y1 = base + y1*dim2
      idx_a = base_y0 + x0
      idx_b = base_y1 + x0
      idx_c = base_y0 + x1
      idx_d = base_y1 + x1

      # use indices to lookup pixels in the flat image and restore
      # channels dim
      im_flat = tf.reshape(im, tf.pack([-1, channel_x]))
      #aa = tf.pack([-1, channels])
      #im_flat = tf.reshape(im, [-1, channels])
      #print im.get_shape(), im_flat.get_shape() #, aa.get_shape()
      im_flat = tf.cast(im_flat, 'float32')
      Ia = tf.gather(im_flat, idx_a)
      Ib = tf.gather(im_flat, idx_b)
      Ic = tf.gather(im_flat, idx_c)
      Id = tf.gather(im_flat, idx_d)
      #print im_flat.get_shape(), idx_a.get_shape()
      #print Ia.get_shape(), Ib.get_shape(), Ic.get_shape(), Id.get_shape()
      # and finally calculate interpolated values
      x0_f = tf.cast(x0, 'float32')
      x1_f = tf.cast(x1, 'float32')
      y0_f = tf.cast(y0, 'float32')
      y1_f = tf.cast(y1, 'float32')
      wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
      wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
      wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
      wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
      #print wa.get_shape(), wb.get_shape(), wc.get_shape(), wd.get_shape()
      output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
      #print output.get_shape()
      return output

  def _meshgrid(self, height, width):
    with tf.variable_scope('_meshgrid'):
      # This should be equivalent to:
      #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
      #                         np.linspace(-1, 1, height))
      #  ones = np.ones(np.prod(x_t.shape))
      #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
      x_t = tf.matmul(tf.ones(shape=tf.pack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
      y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.pack([1, width])))

      x_t_flat = tf.reshape(x_t, (1, -1))
      y_t_flat = tf.reshape(y_t, (1, -1))

      ones = tf.ones_like(x_t_flat)
      grid = tf.concat(0, [x_t_flat, y_t_flat, ones])
      return grid

  def _transform(self, theta, input_dim, out_size, channel_input):
    with tf.variable_scope('_transform'):
      print input_dim.get_shape(), theta.get_shape(), out_size[0], out_size[1]
      num_batch = self.hps.batch_size #tf.shape(input_dim)[0]
      height = tf.shape(input_dim)[1]
      width = tf.shape(input_dim)[2]
      num_channels = tf.shape(input_dim)[3]
      theta = tf.reshape(theta, (-1, 2, 3))
      theta = tf.cast(theta, 'float32')
      
      # grid of (x_t, y_t, 1), eq (1) in ref [1]
      height_f = tf.cast(height, 'float32')
      width_f = tf.cast(width, 'float32')
      out_height = out_size[0]
      out_width = out_size[1]
      grid = self._meshgrid(out_height, out_width)
      #print grid, grid.get_shape()
      grid = tf.expand_dims(grid, 0)
      grid = tf.reshape(grid, [-1])
      grid = tf.tile(grid, tf.pack([num_batch]))
      grid = tf.reshape(grid, tf.pack([num_batch, 3, -1]))
      #print grid, grid.get_shape()

      # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
      T_g = tf.batch_matmul(theta, grid)
      x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
      y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
      x_s_flat = tf.reshape(x_s, [-1])
      y_s_flat = tf.reshape(y_s, [-1])
      #print x_s_flat.get_shape(), y_s_flat.get_shape()
      input_transformed = self._interpolate(input_dim, x_s_flat, y_s_flat, out_size, channel_input)
      #print input_transformed.get_shape()

      output = tf.reshape(input_transformed, tf.pack([num_batch, out_height, out_width, channel_input]))
      return output
      #return input_dim
