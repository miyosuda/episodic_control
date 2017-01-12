# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from environment.environment import Environment
from projection.vae_projection import VAEProjection
from ec.qec_table import QECTable
from agent.agent import EpisodicControlAgent

k = 50
knn_capacity = 100000
observation_dim = 84 * 84
state_dim = 64
gamma = 0.99
epsilon = 0.005
vae_train_step_size = 400000
vae_train_batch_size = 100

CHECKPOINT_DIR = '/tmp/ec_checkpoints'
RECONSTRUCTION_IMAGE_DIR = '/tmp/ec_reconstr_img'
TRAIN_LOG_INTERVAL = 10
TRAIN_SAVE_INTERVAL = 100
TRAIN_SAVE_INTERVAL = 100
RECONSTRUCTION_CHECK_INTERVAL = 100

def train_vae(sess, vae_projection, environment):
  """ Train VAE for projection. """
  saver = tf.train.Saver()
  # Load checkpoint
  step = vae_projection.load(sess, saver, CHECKPOINT_DIR)

  while step < vae_train_step_size:
    loss, reconstr_loss, latent_loss = vae_projection.train(sess,
                                                            environment,
                                                            vae_train_batch_size)

    if step % TRAIN_LOG_INTERVAL == 0:
      print("step={0}, loss={1}, rec_loss={2}, latent_loss={3}".format(step,
                                                                       loss,
                                                                       reconstr_loss,
                                                                       latent_loss))

    step += 1
    
    if step % TRAIN_SAVE_INTERVAL == 0:
      # Save checkpoint
      vae_projection.save(sess, saver, step, CHECKPOINT_DIR)

    if step % RECONSTRUCTION_CHECK_INTERVAL == 0:
      # Create reconstruction image
      vae_projection.check_reconstruction(sess, environment, 10, RECONSTRUCTION_IMAGE_DIR)

def train_episodic_control(agent):
  # TODO:
  for i in range(1):
    ret = agent.step()
    if ret != None:
      print(ret)

num_actions = Environment.get_action_size()
environment = Environment.create_environment()

vae_projection = VAEProjection()

qec_table = QECTable(vae_projection, state_dim, num_actions, k, knn_capacity)

agent = EpisodicControlAgent(environment, qec_table, num_actions, gamma, epsilon)

# Session should be started after Lab environment is created. (To run Lab with GPU)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

vae_projection.set_session(sess)

train_vae(sess, vae_projection, environment)
train_episodic_control(agent)
