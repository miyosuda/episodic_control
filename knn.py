# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import KDTree


class KNN:
  def __init__(self, capacity, state_dim):
    self._capacity = capacity
    self._states = np.zeros((capacity, state_dim))
    self._q_values = np.zeros(capacity)
    self._lru = np.zeros(capacity)
    self._current_capacity = 0
    self._time = 0.0
    self._tree = None

  def peek(self, state):
    index = self._peek_index(state)
    if index >= 0:
      return self._q_values[index]
    return None
    
  def update(self, state, r):
    index = self._peek_index(state)
    if index >= 0:
      # 正確にヒットした場合
      max_q = max(self._q_values[index], r)
      self._q_values[index] = max_q
    else:
      # みつからなかった場合はエントリを追加
      self._add(state, r)
      
  def knn_value(self, state, k):
    if self._current_capacity == 0:
      return 0.0

    _, indices = self._tree.query([state], k=k)

    size = len(indices[0])
    if size == 0:
      return 0.0

    value = 0.0
    for index in indices[0]:
      value += self._q_values[index]
      self._lru[index] = self._time
      self._time += 0.01

    return value / size

  def _peek_index(self, state):
    if self._current_capacity == 0:
      return -1

    # 一番近いものを一つ探してくる.
    _, indices = self._tree.query([state], k=1)
    index = indices[0][0]

    if np.allclose(self._states[index], state):
      # 一番近かったものが、一致していた
      self._lru[index] = self._time
      self._time += 0.01
      return index
    else:
      return -1

  def _add(self, state, r):
    if self._current_capacity >= self._capacity:
      # find the LRU entry
      old_index = np.argmin(self._lru) # リニアサーチしている
      self._states[old_index] = state
      self._q_values[old_index] = r
      self._lru[old_index] = self._time
    else:
      self._states[self._current_capacity] = state
      self._q_values[self._current_capacity] = r
      self._lru[self._current_capacity] = self._time
      self._current_capacity += 1
   
    self._time += 0.01
    # ツリーを作り直し
    self._tree = KDTree(self._states[:self._current_capacity])

    # TDOO: _timeが大きくなった時のラップ処理が必要
