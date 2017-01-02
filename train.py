# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from environment.environment import Environment
from ec.projection import Projection
from ec.qec_table import QECTable
from agent.agent import EpisodicControlAgent

k = 50
knn_capacity = 100000
observation_dim = 84 * 84
state_dim = 64
gamma = 0.99
epsilon = 0.005

num_actions = Environment.get_action_size()
print("action size={0}".format(num_actions))

environment = Environment.create_environment()
projection = Projection(observation_dim, state_dim)

qec_table = QECTable(projection, state_dim, num_actions, k, knn_capacity)

agent = EpisodicControlAgent(environment, qec_table, num_actions, gamma, epsilon)

for i in range(1):
  ret = agent.step()
  if ret != None:
    print(ret)
