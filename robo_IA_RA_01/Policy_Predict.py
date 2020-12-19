# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:28:56 2020

@author: jrmfi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.trajectories import time_step as ts
from comunica import  Comunica
tf.compat.v1.enable_v2_behavior()

# model 3
media = np.array([ 1.35879216e+03,  1.12922585e+00,  2.14542909e+01,  1.92168434e+01,
        5.11025285e+01,  1.02212462e+01,  2.08635187e+01, -1.06422484e+01,
        3.05193599e+01,  7.63876375e+01,  1.00011448e+02,  9.09959047e+03])
std = np.array([2.69301198e+02, 4.90009239e+01, 1.82863123e+01, 1.68678866e+01,
       1.23758189e+01, 6.89640383e+01, 1.15678334e+02, 5.00498245e+01,
       1.26525875e+01, 2.38955187e+01, 1.79462937e-01, 1.65639665e+05])
# batch_size = 3
saved_policy = tf.saved_model.load('policy')
# policy_state = saved_policy.get_initial_state(batch_size=batch_size)

HOST = ''    # Host
PORT = 8888  # Porta
R = Comunica(HOST,PORT)
s = R.createServer()

while True:
    p,addr = R.runServer(s)
    jm = np.array((p-media)/std)
    jm = np.array(jm, dtype=np.float32)
    observations = tf.constant([[jm]])
    # print(observations)
    time_step = ts.restart(observations,1)
    # print(time_step)
    action = saved_policy.action(time_step)
    previsao2 = action.action.numpy()[0]
    d3 = p[0]
    print('recebido: ',p[0])
    print('previsao: ',previsao2)
    if previsao2 == 0:
        print('Sem operacao')
    if previsao2 == 1:
        flag = "compra-{}".format(d3)
        # flag ="compra"
        print('compra: ',previsao2)
        R.enviaDados(flag,s,addr)
    if previsao2 == 2:
        flag = "venda-{}".format(d3)
        # flag = "venda"
        print('venda: ',previsao2)
        R.enviaDados(flag,s,addr)


