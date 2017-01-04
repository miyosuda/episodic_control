# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from scipy.misc import imsave
from projection.vae_projection import VAEProjection
from environment.environment import Environment


CHECKPOINT_DIR = '/tmp/ec_checkpoints'
RECONSTRUCTION_IMAGE_DIR = '/tmp/ec_reconstr_img'

class TestVAEProjection(tf.test.TestCase):
  def test_vae_reconstruct(self):
    environment = Environment.create_environment()
    vae_projection = VAEProjection()

    with self.test_session() as sess:
      init = tf.global_variables_initializer()
      sess.run(init)
      
      vae_projection.set_session(sess)
      
      saver = tf.train.Saver()
      _ = vae_projection.load(sess, saver, CHECKPOINT_DIR)

      vae = vae_projection._vae

      batch_x = []

      for i in range(10):
        observation = environment.random_step()
        observation = observation.reshape((84, 84, 1))
        batch_x.append(observation)
        
      batch_x_reconstr = vae.reconstruct(sess, batch_x)

      if not os.path.exists(RECONSTRUCTION_IMAGE_DIR):
        os.mkdir(RECONSTRUCTION_IMAGE_DIR)

      for i in range(10):
        org_img      = batch_x[i].reshape(84, 84)
        reconstr_img = batch_x_reconstr[i].reshape(84, 84)
        org_img_path = "{0}/img_{1}_org.png".format(RECONSTRUCTION_IMAGE_DIR,i)
        reconstr_img_path = "{0}/img_{1}_rec.png".format(RECONSTRUCTION_IMAGE_DIR,i)
        imsave(org_img_path, org_img)
        imsave(reconstr_img_path, reconstr_img)
