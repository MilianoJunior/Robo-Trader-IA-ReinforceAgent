# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:30:23 2020

@author: jrmfi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
# import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import chardet
import pandas as pd


import tensorflow as tf

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import policy_saver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import q_rnn_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


from CardTrader import CardGameEnv

tf.compat.v1.enable_v2_behavior()

" 1 passo: importar os dados"
try:
    with open('M3.csv', 'rb') as f:
        result = chardet.detect(f.read())  # or readline if the file is large
    
    base = pd.read_csv('M3.csv', encoding=result['encoding'])
    
except:
    print('Erro, é preciso fazer o download dos dados OHLC em csv')


#Hiperparametros

num_day = 565
num_iterations = 2500 # @param {type:"integer"}
collect_episodes_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = 20000 # @param {type:"integer"}

fc_layer_params = (100,)

learning_rate = 1e-3 # @param {type:"number"}
log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 50 # @param {type:"integer"}

train_py_env = CardGameEnv(base,num_day)
eval_py_env = CardGameEnv(base,num_day)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# as we are using dictionary in our enviroment, we will create preprocessing layer

q_net = actor_distribution_network.ActorDistributionNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=q_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
tf_agent.initialize()

# Os agentes contêm duas políticas: a principal política utilizada para avaliação/implantação (agent.policy) e 
# outra política utilizada para coleta de dados (agent.collect_policy)
eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

def collect_episode(environment, policy, num_episodes):

  episode_counter = 0
  environment.reset()

  while episode_counter < num_episodes:
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

    if traj.is_boundary():
      episode_counter += 1


try:
  %%time
except:
  pass

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few episodes using collect_policy and save to the replay buffer.
  collect_episode(
      train_env, tf_agent.collect_policy, collect_episodes_per_iteration)

  # Use data from the buffer and update the agent's network.
  experience = replay_buffer.gather_all()
  train_loss = tf_agent.train(experience)
  replay_buffer.clear()

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)
    
steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim(top=250)


my_policy = eval_policy
saver = policy_saver.PolicySaver(my_policy, batch_size=None)
saver.save('policy')

saved_policy = tf.compat.v2.saved_model.load('policy')

colunas = ['Hora','dif', 'retacao +','retracao -', 'RSI',
             'M22M44', 'M22M66', 'M66M44', 'ADX', 'ATR',
            'Momentum', 'Force']

colunas1 = ['Hora', 'open', 'high', 'low', 'close']
dados1 = pd.DataFrame(data=base[-num_day:-1].values,columns=base.columns)      
dados2 = pd.DataFrame(data=base[-num_day:-1].values,columns=base.columns)
dados1 = dados1[colunas1]
dados2 = dados2[colunas]
index = 0
for i in dados2.values:
    base1 = i[0].split(':')
    dados2.at[index, 'Hora'] = float(base1[0])*100 + float(base1[1])
    index += 1
train_mean = dados2.mean(axis=0)
train_std = dados2.std(axis=0)
dados2 = (dados2 - train_mean) / train_std

from Trade import Trade
from tf_agents.trajectories import time_step as ts

trader = Trade()

import random
stop = -500
gain = 500
trader.reset()
action = 0
for i in range(len(dados1)):
    
    compra,venda,neg,ficha,comprado,vendido,recompensa= trader.agente(dados1.values[i],action,stop,gain,0)
    # print('estado: ',dados2.values[i])
    observations = tf.constant([[dados2.values[i]]])
    time_step = ts.restart(observations,1)
    action2 = saved_policy.action(time_step)
    # time_step = ts.transition(observations,1)
    # action2 = agent.policy.action(time_step)
    action = action2.action.numpy()[0]
    
    print(i,'------------------')
    print('acao: ',action)
    print('comprado: ',comprado)
    print('vendido: ',vendido)
    print('recompensa: ',recompensa)
    
    print('recompensa: ',time_step.reward.numpy(),' action: ',action2.action.numpy()[0])

print(sum(neg.ganhofinal))
