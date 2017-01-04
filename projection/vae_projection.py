# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
from projection.projection import Projection
from projection.vae import VAE
from scipy.misc import imsave


class VAEProjection(Projection):
  def __init__(self):
    self._vae = VAE()

  def set_session(self, sess):
    self._sess = sess

  def project(self, observation):
    x = observation.reshape(1, 84, 84, 1)
    z_mean, z_log_sigma_sq = self._vae.transform(self._sess, x)
    
    z_mean = z_mean.reshape((32))
    z_log_sigma_sq = z_log_sigma_sq.reshape((32))

    state = np.concatenate((z_mean, z_log_sigma_sq))
    return state

  def load(self, sess, saver, checkpoint_dir):
    """ Load checkpoint.

    Returns:
      Current step.
    """
    step = 0
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
      saver.restore(sess, checkpoint.model_checkpoint_path)
      print("Checkpoint loaded: {}".format(checkpoint.model_checkpoint_path))
      tokens = checkpoint.model_checkpoint_path.split("-")
      # set global step
      step = int(tokens[1])
      print("Start from step: {}".format(step))
    else:
      print("Could not find old checkpoint")
    return step

  def save(self, sess, saver, step, checkpoint_dir):
    """ Save checkpoint. """
    if not os.path.exists(checkpoint_dir):
      os.mkdir(checkpoint_dir)
    saver.save(sess, checkpoint_dir + '/' + 'checkpoint', global_step = step)
    print("Checkpoint saved: step={}".format(step))

  def train(self, sess, environment, batch_size):
    """ Load checkpoint.

    Returns:
      Traning loss.
    """
    # Collect 100 images with random actions
    bach_x = []
    for _ in range(batch_size):
      observation = environment.random_step()
      observation = observation.reshape((84, 84, 1))
      bach_x.append(observation)
      
    # Train with mini-batch of 100 images
    loss = self._vae.train(sess, bach_x)
    return loss

  def check_reconstruction(self, sess, environment,
                           image_size,
                           reconstruction_image_dir):
    """ Check VAE reconstruction by reconstructing images. """
    
    if not os.path.exists(reconstruction_image_dir):
      os.mkdir(reconstruction_image_dir)
      
    batch_x = []

    for i in range(image_size):
      observation = environment.random_step()
      observation = observation.reshape((84, 84, 1))
      batch_x.append(observation)
    
    batch_x_reconstr = self._vae.reconstruct(sess, batch_x)
  
    for i in range(image_size):
      org_img      = batch_x[i].reshape(84, 84)
      reconstr_img = batch_x_reconstr[i].reshape(84, 84)
      org_img_path      = "{0}/img_{1}_org.png".format(reconstruction_image_dir,i)
      reconstr_img_path = "{0}/img_{1}_rec.png".format(reconstruction_image_dir,i)
      imsave(org_img_path, org_img)
      imsave(reconstr_img_path, reconstr_img)

    print("Reconstruction images created.")

