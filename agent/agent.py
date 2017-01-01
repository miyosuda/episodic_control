# -*- coding: utf-8 -*-
import numpy as np


class Frame(object):
  def __init__(self, observation, action, reward):
    self.observation = observation
    self.action = action
    self.reward = reward


class FrameBuffer(object):
  def __init__(self):
    self.frames = []

  def add_frame(self, observation, action, reward):
    frame = Frame(observation, action, reward)
    self.frames.append(frame)


class EpisodicControlAgent(object):
  def __init__(self, environment, qec_table, num_actions):
    self._environment = environment
    self._qec_table = qec_table    
    self._gamma = 0.99
    self._epsilon = 0.005
    self._num_actions = num_actions

    self._reset()
    
  def _reset(self):
    self._frame_buffer = FrameBuffer()
    self._episode_reward = 0
  
  def step(self):
    """
    Returns:
      Episode reward when episode ends. Otherwise None.
    """
    last_observation = self._environment.last_observation
    # Choose action based on QEC table
    action = self._choose_action(last_observation)
    _, reward, terminal = self._environment.step(action)

    # Record frame
    self._frame_buffer.add_frame(last_observation,
                                 action,
                                 np.clip(reward, -1, 1))
    
    self._episode_reward += reward

    # Update QEC table when episode ends
    if terminal:
      self._update_qec_table()
      episode_reward = self._episode_reward
      self._reset()
      return episode_reward
    else:
      return None

  def _choose_action(self, observation):
    # epsilon greedy
    if np.random.rand() < self._epsilon:
      return np.random.randint(0, self._num_actions)
    else:
      return self._qec_table.get_max_qec_action(observation)

  def _update_qec_table(self):
    """ Update QEC table """
    R = 0.0
    # len-1から0まで降順
    for i in range(len(self._frame_buffer.frames)-1, -1, -1):
      frame = self._frame_buffer.frames[i]
      # discountしていく
      R = R * self._gamma + frame.reward
      # 求めたQEC値で、QECテーブルの値を更新
      # (エントリにヒットしたら値を更新し、ヒットしない場合はエントリを追加)
      self._qec_table.update(frame.observation, frame.action, R)
