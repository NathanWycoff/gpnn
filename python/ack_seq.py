#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/ack_seq.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.19.2019

## Do sequential design on the Ackley function.
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
tf.enable_eager_execution()
from scipy.optimize import minimize
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.ion()
from scipy.special import expit, logit
from scipy.spatial import distance_matrix
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
exec(open("python/hilbert_curve.py").read())
exec(open("python/ackley.py").read())
exec(open("python/neural_maxent.py").read())
exec(open("python/seq_des.py").read())
exec(open("python/opt_lib.py").read())
exec(open("python/misc.py").read())
exec(open("python/test_funcs.py").read())

tf.random.set_random_seed(1234)
np.random.seed(1234)

N_init = 20
P = 20
L = 1
H = 10
R = 2
seq_steps = 30

# Two similarly shaped random nets, one is the one we init on, one the one we use.
used_model = random_nn(P,L,H,R, act = tf.nn.tanh)

init_w = used_model.get_weights()
design = neural_maxent(N_init ,P, L, H, R, net_weights = used_model.get_weights())['design']
#true_extent = get_extent(design, true_model)

bb_obj = lambda x: ackley(x.flatten())

response_us = np.apply_along_axis(bb_obj, 1, design)
y_mu = np.mean(response_us)
y_sig = np.std(response_us)
response = (response_us - y_mu) / y_sig

# Update model after initial design.
est_model = update_weights(design, response, used_model, l2_coef = 0)

for si in range(seq_steps):
    # Sample a next point; append it to our design
    next_point = neural_maxent(1, P, L, H, R, X_prev = design, net_weights = est_model.get_weights())['design']
    next_y = (bb_obj(next_point) - y_mu) / y_sig

    design = np.vstack([design, next_point])
    response = np.append(response, next_y)

    # Update our neural net weights.
    #TODO: Does this converge? Do, eventually, we learn a W and stop learning mutch upon a new point being revealed?
    est_model = update_weights(design, response, est_model, l2_coef = 0)
