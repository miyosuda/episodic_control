# -*- coding: utf-8 -*-
import numpy as np
from environment import Environment
from projection import Projection
from qec_table import QECTable
from agent import EpisodicControlAgent

k = 11
knn_capacity = 1000000 # Labirynthでは 100000
observation_dim = 84 * 84
state_dim = 64 # Labirynthでは32
gamma = 1.0

num_actions = Environment.get_action_size()
environment = Environment.create_environment()
projection = Projection(observation_dim, state_dim)

qec_table = QECTable(projection, state_dim, num_actions, k, knn_capacity)

agent = EpisodicControlAgent(environment, qec_table, num_actions)

for i in range(100000):
  ret = agent.step()
  if ret != None:
    print(ret)
