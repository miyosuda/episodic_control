# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
#import gym
import deepmind_lab


#GYM_ENV = 'Breakout-v0'

class Environment(object):
  # cached action size
  action_size = -1
  
  @staticmethod
  def create_environment():
    return LabEnvironment()
    #return GymEnvironment()
  
  @staticmethod
  def get_action_size():
    if Environment.action_size >= 0:
      return Environment.action_size

    Environment.action_size = 8
    return Environment.action_size
    """
    env = gym.make(GYM_ENV)
    Environment.action_size = env.action_space.n
    print("intialize action size={0}".format(Environment.action_size))
    return Environment.action_size
    """

  def __init__(self):
    pass

  def process(self, action):
    pass

  def reset(self):
    pass


"""
class GymEnvironment(Environment):
  def __init__(self, display=False, frame_skip=4):
    Environment.__init__(self)
    
    self._display = display
    self._frame_skip = frame_skip
    if self._frame_skip < 1:
      self._frame_skip = 1
    
    self.env = gym.make(GYM_ENV)
    self.reset()
    
  def reset(self):
    obs = self.env.reset()
    self.last_observation = self._preprocess_frame(obs)
    
  def _preprocess_frame(self, image):
    # image shape = (210, 160, 3)
    image = image.astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized_image = cv2.resize(image, (84, 84))
    resized_image = resized_image / 255.0
    return resized_image
  
  def step(self, action):
    if self._display:
      self.env.render()

    reward = 0
    for i in range(self._frame_skip):
      obs, r, terminal, _ = self.env.step(action)
      reward += r
      if terminal:
        break
    self.last_observation = self._preprocess_frame(obs)
    return self.last_observation, reward, terminal
"""

def _action(*entries):
  return np.array(entries, dtype=np.intc)

class LabEnvironment(Environment):
  ACTIONS = {
      'look_left':    _action(-20,   0,  0,  0, 0, 0, 0),
      'look_right':   _action( 20,   0,  0,  0, 0, 0, 0),
      'look_up':      _action(  0,  10,  0,  0, 0, 0, 0),
      'look_down':    _action(  0, -10,  0,  0, 0, 0, 0),
      'strafe_left':  _action(  0,   0, -1,  0, 0, 0, 0),
      'strafe_right': _action(  0,   0,  1,  0, 0, 0, 0),
      'forward':      _action(  0,   0,  0,  1, 0, 0, 0),
      'backward':     _action(  0,   0,  0, -1, 0, 0, 0),
      'fire':         _action(  0,   0,  0,  0, 1, 0, 0),
      'jump':         _action(  0,   0,  0,  0, 0, 1, 0),
      'crouch':       _action(  0,   0,  0,  0, 0, 0, 1)
  }

  ACTION_LIST = ACTIONS.values()
  
  def __init__(self):
    Environment.__init__(self)

    level = 'seekavoid_arena_01'

    self._env = deepmind_lab.Lab(
      level,
      ['RGB_INTERLACED'],
      config={
        'fps': str(60),
        'width': str(84),
        'height': str(84)
      })
    self.reset()

  def reset(self):
    self._env.reset()
    obs = self._env.observations()['RGB_INTERLACED']
    self.last_observation = self._preprocess_frame(obs)
    
  def _preprocess_frame(self, image):
    image = image.astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.reshape((84,84))
    image = image / 255.0
    return image

  def step(self, action):
    real_action = LabEnvironment.ACTION_LIST[action]
    
    reward = self._env.step(real_action, num_steps=4)
    terminal = not self._env.is_running()

    obs = self._env.observations()['RGB_INTERLACED']
    self.last_observation = self._preprocess_frame(obs)
    return self.last_observation, reward, terminal
