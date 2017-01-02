# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import scipy.misc
import sys
from environment.environment import Environment

SAVE_IMAGE = False

class TestGameEnvironment(unittest.TestCase):
  def test_step(self):
    environment = Environment.create_environment()
    action_size = Environment.get_action_size()

    if sys.platform == 'darwin':
      self.assertTrue( action_size == 6 )
    else:
      self.assertTrue( action_size == 8 )

    for i in range(3):
      self.assertTrue( environment.last_observation.shape == (84,84) )
      if SAVE_IMAGE:
        scipy.misc.imsave("debug_observation{0}.png".format(i), environment.last_observation)
      reward, terminal = environment.step(0)


  def test_random_step(self):
    environment = Environment.create_environment()

    for i in range(3):
      observation = environment.random_step()
      self.assertTrue( observation.shape == (84,84) )

if __name__ == '__main__':
  unittest.main()
