# -*- coding: utf-8 -*-
import numpy as np
from knn import KNN

class QECTable(object):
  def __init__(self,
               k,
               state_dim,
               observation_dim,
               knn_capacity,      # knnのバッファサイズ
               num_actions,
               rng):
    
    self._k = k
    self._rng = rng

    self._ec_buffer = [] # リストのエントリ数は、アクション数    
    for i in range(num_actions):
      knn = KNN(knn_capacity, state_dim)
      self._ec_buffer.append(knn)

    # ランラムprojection
    self._matrix_projection = self._rng.randn(state_dim,
                                              observation_dim).astype(np.float32)

  def estimate(self, observation, action):
    # projectionする
    state = np.dot(self._matrix_projection, observation.flatten())

    # バッファからヒットするものを探してくる
    q = self._ec_buffer[a].peek(state)
    if q != None:
      # ヒットしたらそのQEC値を返す
      return q

    # ヒットしない場合はKNNで平均
    return self._ec_buffer[action].knn_value(state, self._k)

  # s is 84 * 84 * 3
  # a is 0 to num_actions
  # r is reward
  def update(self, observation, action, r):
    # projectionする
    state = np.dot(self._matrix_projection, observation.flatten())
    
    # バッファからヒットするものを探し、ヒットしたら値を更新, ヒットしなかったら
    # エントリを追加
    self._ec_buffer[action].update(state, r)


class Frame(object):
  def __init__(self, observation, action, reward, terminal):
    self.observation = observation
    self.action = action
    self.reward = reward
    self.terminal = terminal


class FrameBuffer(object):
  def __init__(self):
    self.frames = []

  def add_frame(self, observation, action, reward, terminal):
    frame = Frame(observation, action, reward, terminal)
    self.frames.append(frame)


class EpisodicControl(object):
  def __init__(self):
    self._gamma = 0.99
    self._epsilon = 1.0
    self._epsilon_min = 0.005
    self._epsilon_rate = ***
    self._rng = ***
    self._num_actions = ***
        
  def start_episode(self, observation):
    self._episode_reward = 0
    self._frame_buffer = FrameBuffer()
    
    action = self._rng.randint(0, self._num_actions)

    self._last_action = action
    self._last_observation = observation
    return action
  
  def _choose_action(self, observation, reward):
    # フレームを記録 (actionは前回のものを利用)
    self._frame_buffer.add_frame(self._last_observation,
                                 self._last_action,
                                 reward,
                                 False)

    # epsilon greedy
    if self._rng.rand() < self._epsilon:
      return self._rng.randint(0, self._num_actions)

    # 最大のQECのactionを探してきて返す
    value = -100
    max_action = 0
    
    # argmax(Q(s,a))
    for action in range(self._num_actions):
      # QECテーブルを元に、state, actionをからQEC値を推定する
      value_t = self._qec_table.estimate(observation, action)
      if value_t > value:
        value = value_t
        max_action = action

    return max_action

  def step(self, reward, observation):
    self._episode_reward += reward

    # epsilonをstepが進むごとに下げていく
    self._epsilon = max(self._epsilon_min, self._epsilon - self._epsilon_rate)

    # EC tableを元にactionを選択する
    action = self._choose_action(observation,
                                 np.clip(reward, -1, 1))

    self._last_action = action
    self._last_observation = observation
    return action
  
  def end_episode(self, reward, terminal=True):
    self._episode_reward += reward
    
    # フレームを記録
    self._frame_buffer.add_frame(self.last_observation,
                                 self.last_action,
                                 np.clip(reward, -1, 1),
                                 True)

    # QECテーブルを更新
    R = 0.0
    # len-1から0まで降順
    for i in range(len(self.frame_buffer.frames)-1, -1, -1):
      frame = self.frame_buffer.frames[i]
      # discountしていく
      R = R * self._gamma + frame.reward
      # 求めたQEC値で、QECテーブルの値を更新
      # (エントリにヒットしたら値を更新し、ヒットしない場合はエントリを追加)
      self.qec_table.update(frame.observation, frame.action, R)
