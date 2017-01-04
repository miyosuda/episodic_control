# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from projection.vae_projection import VAEProjection


class TestVAEProjection(tf.test.TestCase):
  def test_project(self):
    vae_projection = VAEProjection()

    with self.test_session() as sess:
      init = tf.global_variables_initializer()
      sess.run(init)
      
      vae_projection.set_session(sess)

      observation = np.zeros((84,84))

      state = vae_projection.project(observation)
      self.assertTrue( state.shape == (64,) )


if __name__ == "__main__":
  tf.test.main()
