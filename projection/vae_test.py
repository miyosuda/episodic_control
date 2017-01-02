# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from projection.vae import VAE


class TestVAE(tf.test.TestCase):
  def _check_shape(self, node, dims):
    shape = node.get_shape()
    self.assertTrue(node.get_shape().as_list() == dims)
  
  def test_node_shape(self):
    vae = VAE()

    # Checking node shapes
    self._check_shape(vae.z_mean,         [None, 32])
    self._check_shape(vae.z_log_sigma_sq, [None, 32])
    self._check_shape(vae.z,              [None, 32])

    self._check_shape(vae.x_reconstr_mean,         [None, None, None, None])
    self._check_shape(vae.x_reconstr_log_sigma_sq, [None, None, None, None])
    
    self._check_shape(vae.loss, [])

    with self.test_session() as sess:
      # Test train()
      init = tf.global_variables_initializer()
      sess.run(init)

      x = np.zeros((1,84,84,1))
      _ = vae.train(sess, x)

      # Test transform()
      z_mean, z_log_sigma_sq = vae.transform(sess, x)
      # Check shape of latent variables
      self.assertTrue(z_mean.shape == (1,32))
      self.assertTrue(z_log_sigma_sq.shape == (1,32))

      # Test generate()
      x_gen = vae.generate(sess)
      # Check shape of generated image
      self.assertTrue(x_gen.shape == (1,84,84,1))

      # Test reconstruct()
      x_reconstr = vae.reconstruct(sess, x)
      # Check shape of reconstructed image
      self.assertTrue(x_reconstr.shape == (1,84,84,1))

    
if __name__ == '__main__':
  tf.test.main()  
