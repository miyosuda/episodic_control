# -*- coding: utf-8 -*-
import unittest
import numpy as np
import scipy.misc

from environment import Environment

class TestGameEnvironment(unittest.TestCase):
  def test_step(self):
    environment = Environment.create_environment()
    action_size = Environment.get_action_size()

    print("action_size={}".format(action_size))

    for i in range(3):
      observation, reward, terminal = environment.step(0)
      print(observation.shape)
      self.assertTrue( observation.shape == (84,84) )
      #scipy.misc.imsave("debug_observation{0}.png".format(i), observation)

if __name__ == '__main__':
  unittest.main()
