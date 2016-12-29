# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import gym
import cv2

# TODO: 引数で与える
GYM_ENV = 'Breakout-v0'

class Environment(object):
  # cached action size
  action_size = -1
  
  @staticmethod
  def create_environment():
    return GameEnvironment()
  
  @staticmethod
  def get_action_size():
    if Environment.action_size >= 0:
      return Environment.action_size

    env = gym.make(GYM_ENV)
    Environment.action_size = env.action_space.n
    print("intialize action size={0}".format(Environment.action_size))
    return Environment.action_size

  def __init__(self):
    pass

  def process(self, action):
    pass

  def reset(self):
    pass


class GameEnvironment(Environment):
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
    last_observation = self._preprocess_frame(obs)
    return last_observation, reward, terminal
