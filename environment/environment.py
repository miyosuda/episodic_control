# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np


USE_LAB = True
if sys.platform == 'darwin':
  # On MacOSX, use gym instead of deepmind lab for test
  USE_LAB = False


class Environment(object):
  # cached action size
  action_size = -1
  
  @staticmethod
  def create_environment():
    if USE_LAB:
      from . import lab_environment
      return lab_environment.LabEnvironment()
    else:
      from . import gym_environment
      return gym_environment.GymEnvironment()
  
  @staticmethod
  def get_action_size():
    if Environment.action_size >= 0:
      return Environment.action_size
    
    if USE_LAB:
      from . import lab_environment
      Environment.action_size = \
        lab_environment.LabEnvironment.get_action_size()
    else:
      from . import gym_environment
      Environment.action_size = \
        gym_environment.GymEnvironment.get_action_size()
    return Environment.action_size

  def __init__(self):
    pass

  def step(self, action):
    pass

  def reset(self):
    pass

  def random_step(self):
    num_actions = Environment.get_action_size()
    action = np.random.randint(num_actions)
    observation = self.last_observation
    reward, terminal = self.step(action)
    if terminal:
      self.reset()
    return observation
