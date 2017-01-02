# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf


class VAE(object):
  def __init__(self, learning_rate=1e-5):
    self._learning_rate = learning_rate
    
    self._create_network()
    self._create_loss_optimizer()

  def _conv_weight_variable(self, weight_shape, deconv=False):
    w = weight_shape[0]
    h = weight_shape[1]
    if deconv:
      input_channels  = weight_shape[3]
      output_channels = weight_shape[2]
    else:
      input_channels  = weight_shape[2]
      output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]
    
    weight_initial = tf.random_uniform(weight_shape, minval=-d, maxval=d)
    bias_initial   = tf.random_uniform(bias_shape,   minval=-d, maxval=d)
    return tf.Variable(weight_initial), tf.Variable(bias_initial)
  
  def _fc_weight_variable(self, weight_shape):
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]
    weight_initial = tf.random_uniform(weight_shape, minval=-d, maxval=d)
    bias_initial   = tf.random_uniform(bias_shape,   minval=-d, maxval=d)
    return tf.Variable(weight_initial), tf.Variable(bias_initial)
  
  def _get2d_deconv_output_size(self, input_height, input_width, filter_height,
                                filter_width, row_stride, col_stride, padding_type):
    if padding_type == 'VALID':
      out_height = (input_height - 1) * row_stride + filter_height
      out_width  = (input_width  - 1) * col_stride + filter_width
      
    elif padding_type == 'SAME':
      out_height = input_height * row_stride
      out_width  = input_width  * col_stride
  
    return out_height, out_width
  
  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],
                        padding='VALID')
  
  def _deconv2d(self, x, W, input_width, input_height, stride):
    filter_height = W.get_shape()[0].value
    filter_width  = W.get_shape()[1].value
    out_channel   = W.get_shape()[2].value
    
    out_height, out_width = self._get2d_deconv_output_size(input_height,
                                                           input_width,
                                                           filter_height,
                                                           filter_width,
                                                           stride,
                                                           stride,
                                                           'VALID')
    batch_size = tf.shape(x)[0]
    output_shape = tf.pack([batch_size, out_height, out_width, out_channel])
    return tf.nn.conv2d_transpose(x, W, output_shape,
                                  strides=[1, stride, stride, 1],
                                  padding='VALID')
  
  def _create_recognition_network(self, x):
    # [filter_height, filter_width, in_channels, out_channels]
    W_conv1, b_conv1 = self._conv_weight_variable([4, 4,  1, 32])
    W_conv2, b_conv2 = self._conv_weight_variable([5, 5, 32, 32])
    W_conv3, b_conv3 = self._conv_weight_variable([5, 5, 32, 64])
    W_conv4, b_conv4 = self._conv_weight_variable([4, 4, 64, 64])
    
    W_fc1, b_fc1 = self._fc_weight_variable([3 * 3 * 64, 512])
    W_fcm, b_fcm = self._fc_weight_variable([512, 32])
    W_fcs, b_fcs = self._fc_weight_variable([512, 32])
    
    h_conv1 = tf.nn.relu(self._conv2d(x,       W_conv1, 2) + b_conv1)  # (41, 41)
    h_conv2 = tf.nn.relu(self._conv2d(h_conv1, W_conv2, 2) + b_conv2)  # (19, 19)
    h_conv3 = tf.nn.relu(self._conv2d(h_conv2, W_conv3, 2) + b_conv3)  # (8,  8)
    h_conv4 = tf.nn.relu(self._conv2d(h_conv3, W_conv4, 2) + b_conv4)  # (3,  3)
    h_conv4_flat = tf.reshape(h_conv4, [-1, 3 * 3 * 64])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
    
    z_mean         = tf.tanh(tf.matmul(h_fc1, W_fcm) + b_fcm)
    z_log_sigma_sq = tf.tanh(tf.matmul(h_fc1, W_fcs) + b_fcs)
    return (z_mean, z_log_sigma_sq)

  def _create_generator_network(self, z):
    W_fc1, b_fc1 = self._fc_weight_variable([32, 512])
    W_fc2, b_fc2 = self._fc_weight_variable([512, 3*3*64])
  
    # [filter_height, filter_width, output_channels, in_channels]
    W_deconv1, b_deconv1 = self._conv_weight_variable([4, 4, 64, 64], deconv=True)
    W_deconv2, b_deconv2 = self._conv_weight_variable([5, 5, 32, 64], deconv=True)
    W_deconv3, b_deconv3 = self._conv_weight_variable([5, 5, 32, 32], deconv=True)
    
    W_deconv4_mean,         b_deconv4_mean         = self._conv_weight_variable([4, 4, 1, 32],  deconv=True)
    W_deconv4_log_sigma_sq, b_deconv4_log_sigma_sq = self._conv_weight_variable([4, 4, 1, 32],  deconv=True)
    
    h_fc1 = tf.nn.relu(tf.matmul(z,     W_fc1) + b_fc1) # (-1, 512)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2) # (512, 3*3*64)
    h_fc2_reshaped = tf.reshape(h_fc2, [-1, 3, 3, 64])
    h_deconv1 = tf.nn.relu(self._deconv2d(h_fc2_reshaped, W_deconv1,  3,  3, 2) + b_deconv1)
    h_deconv2 = tf.nn.relu(self._deconv2d(h_deconv1,      W_deconv2,  8,  8, 2) + b_deconv2)
    h_deconv3 = tf.nn.relu(self._deconv2d(h_deconv2,      W_deconv3, 19, 19, 2) + b_deconv3)
    
    x_reconstr_mean         = tf.sigmoid(self._deconv2d(h_deconv3, W_deconv4_mean,         41, 41, 2) + b_deconv4_mean)
    x_reconstr_log_sigma_sq = tf.tanh(   self._deconv2d(h_deconv3, W_deconv4_log_sigma_sq, 41, 41, 2) + b_deconv4_log_sigma_sq)
    return x_reconstr_mean, x_reconstr_log_sigma_sq

  def _sample_z(self, z_mean, z_log_sigma_sq):
    eps_shape = tf.shape(z_mean)
    eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
    # z = mu + sigma * epsilon
    z = tf.add(z_mean,
               tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
    return z

  def _create_network(self):
    # Image input
    self.x = tf.placeholder("float", shape=[None, 84, 84, 1])
      
    # Encoderの出力が、"z_mean"と"z_log_sigma_sq"
    self.z_mean, self.z_log_sigma_sq = self._create_recognition_network(self.x)
    
    # z = mu + sigma * epsilon
    self.z = self._sample_z(self.z_mean, self.z_log_sigma_sq)
    
    # Decoderの出力がx_reconstr_mean, x_reconstr_sigma_sq
    self.x_reconstr_mean, self.x_reconstr_log_sigma_sq = self._create_generator_network(self.z)
    
  def _create_loss_optimizer(self):
    # Reconstruction loss
    reconstr_loss = tf.reduce_sum( (0.5/tf.exp(self.x_reconstr_log_sigma_sq) * tf.square(self.x - self.x_reconstr_mean)) +
                                   (0.5 * math.log(2.0 * math.pi) + 0.5 * self.x_reconstr_log_sigma_sq),
                                   [1,2,3])
    
    # Latent loss
    beta = 1.0
    latent_loss = beta * -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                              - tf.square(self.z_mean) 
                                              - tf.exp(self.z_log_sigma_sq), 1)
    
    self.loss = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch

    self.optimizer = tf.train.RMSPropOptimizer(
      learning_rate=self._learning_rate).minimize(self.loss)

  def train(self, sess, X):
    """Train with image data mini-batch.
    
    Returns:
      Loss of mini-batch.
    """
    _, loss = sess.run((self.optimizer, self.loss), 
                       feed_dict={self.x: X})
    return loss

  def transform(self, sess, X):
    """Transform X into latent space.

    Returns:
      Z mean and Z variance log
    """
    return sess.run( [self.z_mean, self.z_log_sigma_sq],
                     feed_dict={self.x: X})

  def generate(self, sess):
    """ Generate X with Z prior. """
    # Sample with Gaussian with mean=0, variance=1
    z_mu = np.random.normal(size=(1,32))
    
    # Not sampling output with Gaussian.
    return sess.run(self.x_reconstr_mean, 
                    feed_dict={self.z: z_mu})

  def reconstruct(self, sess, X):
    """ Reconstruct X """    
    # Not sampling output with Gaussian.    
    return sess.run(self.x_reconstr_mean,
                    feed_dict={self.x: X})
