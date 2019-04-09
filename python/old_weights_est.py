#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/weights_est.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.03.2019

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
tf.enable_eager_execution()
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.ion()
from scipy.special import expit, logit
from scipy.spatial import distance_matrix
exec(open("python/hilbert_curve.py").read())
exec(open("python/ackley.py").read())
exec(open("python/neural_maxent.py").read())
exec(open("python/test_functions.py").read())
exec(open("python/opt_lib.py").read())

## Sequential design of an acquisition function using a deep kernel.
## A toy design: start with some high D function, map it to R D using a MLP, 
## then do sequential design pretending like we know the true mapping. 
## We should demolish methods which do not avail themselves of this info.
N_init = 20
P = 5
L = 1
H = 10
R = 2
# TODO: These next two dials are pretty fragile.
minalldist = 1e-5
scalealldist = 1500

act = tf.nn.tanh
model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(H, activation=act, input_shape=(P,)) if i == 0 else tf.keras.layers.Dense(H, activation=act) for i in range(L)] + 
    [tf.keras.layers.Dense(R)
    ])
model.build(input_shape=[P])
model.summary()

## Get the extent of the design with many points. TODO: Betterrr
#TODO: WARNING -- ackley_obj uses global variable extent (terrible practice).
viz_design = neural_maxent(100, P, L, H, R, net_weights = model.get_weights())['design']
viz_Z = model(tf.cast(viz_design, tf.float32)).numpy()
extent = [min(viz_Z[:,0]), max(viz_Z[:,0]), min(viz_Z[:,1]), max(viz_Z[:,1])]

# Entropy max initial design
#TODO: WARNING -- ackley_obj uses global variable extent (terrible practice). (DUPLICATE)
N = N_init
design = neural_maxent(N ,P, L, H, R, net_weights = model.get_weights())['design']
response = np.apply_along_axis(ackley_obj, 1, design)

init_w = model.get_weights()
model = update_weights(design, response, model)
opt_w = model.get_weights()

model.set_weights(init_w)
model(tf.cast(design, np.float32))
model.set_weights(opt_w)
model(tf.cast(design, np.float32))