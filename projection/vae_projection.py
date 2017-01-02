# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
from projection.projection import Projection
from projection.vae import VAE


TRAIN_LOG_INTERVAL = 10
TRAIN_SAVE_INTERVAL = 100
CHECKPOINT_DIR = 'checkpoints'


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

  def _load(self, sess, saver):
    step = 0
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
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

  def _save(self, sess, saver, step):
    if not os.path.exists(CHECKPOINT_DIR):
      os.mkdir(CHECKPOINT_DIR)
    saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = step)
    print("Checkpoint saved: step={}".format(step))

  def train(self, sess, environment, train_step_size, batch_size):
    saver = tf.train.Saver()
    # Load checkpoint
    step = self._load(sess, saver)

    while step < train_step_size:
      # Collect 100 images with random actions
      bach_x = []
      for _ in range(batch_size):
        observation = environment.random_step()
        observation = observation.reshape((84, 84, 1))
        bach_x.append(observation)
      # Train with mini-batch of 100 images
      loss = self._vae.train(sess, bach_x)

      if step % TRAIN_LOG_INTERVAL == 0:
        print("step={0}, loss={1}".format(step, loss))
      
      step += 1
      if step % TRAIN_SAVE_INTERVAL == 0:
        # Save checkpoint
        self._save(sess, saver, step)
