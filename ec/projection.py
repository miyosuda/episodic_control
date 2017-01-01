# -*- coding: utf-8 -*-
import numpy as np

RANDOM_SEED = 1

class Projection(object):
  def __init__(self, observation_dim, state_dim):
    # Random projection
    random_state = np.random.RandomState(RANDOM_SEED)
    self._matrix_projection = random_state.randn(state_dim,
                                                 observation_dim).astype(np.float32)

  def project(self, observation):
    state = np.dot(self._matrix_projection, observation.flatten())
    return state
