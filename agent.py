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


class EpisodicControl(object, qec_table, num_actions):
  def __init__(self):
    self._qec_table = qec_table    
    self._gamma = 0.99
    self._epsilon = 0.005
    self._num_actions = num_actions
    
  def start_episode(self, observation):
    self._episode_reward = 0
    self._frame_buffer = FrameBuffer()

    action = 0
    self._last_action = action
    self._last_observation = observation
    return action
  
  def step(self, reward, observation):
    self._episode_reward += reward

    # Choose action based on QEC table
    action = self._choose_action(observation,
                                 np.clip(reward, -1, 1))

    self._last_action = action
    self._last_observation = observation
    return action
  
  def end_episode(self, reward):
    self._episode_reward += reward
    
    # record frame
    self._frame_buffer.add_frame(self.last_observation,
                                 self.last_action,
                                 np.clip(reward, -1, 1))

    # Update QEC table
    R = 0.0
    # len-1から0まで降順
    for i in range(len(self.frame_buffer.frames)-1, -1, -1):
      frame = self.frame_buffer.frames[i]
      # discountしていく
      R = R * self._gamma + frame.reward
      # 求めたQEC値で、QECテーブルの値を更新
      # (エントリにヒットしたら値を更新し、ヒットしない場合はエントリを追加)
      self.qec_table.update(frame.observation, frame.action, R)

  def _choose_action(self, observation, reward):
    # フレームを記録 (actionは前回のものを利用)
    self._frame_buffer.add_frame(self._last_observation,
                                 self._last_action,
                                 reward)

    # epsilon greedy
    if np.random.rand() < self._epsilon:
      return np.random.randint(0, self._num_actions)

    # 最大のQECのactionを探してきて返す
    value = float("-inf")
    max_action = 0
    
    # argmax(Q(s,a))
    for action in range(self._num_actions):
      # QECテーブルを元に、state, actionをからQEC値を推定する
      value_t = self._qec_table.estimate(observation, action)
      if value_t > value:
        value = value_t
        max_action = action

    return max_action
