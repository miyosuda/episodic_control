# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

USE_LAB = False

class Environment(object):
  # cached action size
  action_size = -1
  
  @staticmethod
  def create_environment():
    if USE_LAB:
      import lab_environment
      return lab_environment.LabEnvironment()
    else:
      import gym_environment
      return gym_environment.GymEnvironment()
  
  @staticmethod
  def get_action_size():
    if Environment.action_size >= 0:
      return Environment.action_size
    
    if USE_LAB:
      import lab_environment
      Environment.action_size = \
        lab_environment.LabEnvironment.get_action_size()
    else:
      import gym_environment
      Environment.action_size = \
        gym_environment.GymEnvironment.get_action_size()
    return Environment.action_size

  def __init__(self):
    pass

  def process(self, action):
    pass

  def reset(self):
    pass
