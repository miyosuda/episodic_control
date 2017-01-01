# -*- coding: utf-8 -*-
import numpy as np


class Projection(object):
  # TODO: 乱数を固定にしないとダメか
  def __init__(self, observation_dim, state_dim):
    # Random projection
    self._matrix_projection = np.random.randn(state_dim,
                                              observation_dim).astype(np.float32)

  def project(self, observation):
    state = np.dot(self._matrix_projection, observation.flatten())
    return state
