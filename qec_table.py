# -*- coding: utf-8 -*-
import numpy as np
from knn import KNN


class QECTable(object):
  def __init__(self,
               projection,
               state_dim,
               num_actions,
               k,
               knn_capacity):  # KNN buffer size
    self._k = k
    self._projection = projection

    self._knns = [] # One KNN for each action
    for i in range(num_actions):
      knn = KNN(knn_capacity, state_dim)
      self._knns.append(knn)

  def estimate(self, observation, action):
    state = self._projection.project(observation)

    # バッファからヒットするものを探してくる
    q = self._knns[action].peek(state)
    if q != None:
      # ヒットしたらそのQEC値を返す
      return q

    # ヒットしない場合はKNNで平均
    return self._knns[action].knn_value(state, self._k)

  def update(self, observation, action, r):
    state = self._projection.project(observation)
    
    # バッファからヒットするものを探し、ヒットしたら値を更新, ヒットしなかったら
    # エントリを追加
    self._knns[action].update(state, r)
