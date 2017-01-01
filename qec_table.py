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
    self._num_actions = num_actions

    self._knns = [] # One KNN for each action
    for i in range(num_actions):
      knn = KNN(knn_capacity, state_dim)
      self._knns.append(knn)

  def _estimate(self, state, action):
    # バッファからヒットするものを探してくる
    q = self._knns[action].peek(state)
    if q != None:
      # ヒットしたらそのQEC値を返す
      return q

    # ヒットしない場合はKNNで平均
    return self._knns[action].knn_value(state, self._k)

  def get_max_qec_action(self, observation):
    state = self._projection.project(observation)

    # 最大のQECのactionを探してきて返す
    q = float("-inf")
    max_action = 0
    
    # argmax(Q(s,a))
    for action in range(self._num_actions):
      # QECテーブルを元に、state, actionをからQEC値を推定する
      q_t = self._estimate(state, action)
      if q_t > q:
        q = q_t
        max_action = action

    return max_action

  def update(self, observation, action, r):
    state = self._projection.project(observation)
    
    # バッファからヒットするものを探し、ヒットしたら値を更新, ヒットしなかったら
    # エントリを追加
    self._knns[action].update(state, r)
