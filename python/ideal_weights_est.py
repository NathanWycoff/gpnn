#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/ideal_weights_est.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.09.2019

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

N_init = 10
P = 3
L = 1
H = 10
R = 2
seq_steps = 30

# Two similarly shaped random nets, one is the one we init on, one the one we use.
used_model = random_nn(P,L,H,R, act = tf.nn.tanh)
true_model = random_nn(P,L,H,R, act = tf.nn.tanh)

init_w = used_model.get_weights()
design = neural_maxent(N_init ,P, L, H, R, net_weights = used_model.get_weights())['design']
true_extent = get_extent(design, true_model)

bb_obj = lambda x: neural_ackley(x, model = true_model, extent = true_extent)

response_us = np.apply_along_axis(bb_obj, 1, design)
y_mu = np.mean(response_us)
y_sig = np.std(response_us)
response = (response_us - y_mu) / y_sig

est_model = update_weights(design, response, used_model, l2_coef = 0)

neural_plot(design, response, used_model, bb_obj, figname = 'images/ackley_before.pdf')
neural_plot(design, response, est_model, bb_obj, figname = 'images/ackley_after.pdf')
neural_plot(design, response, true_model, bb_obj, figname = 'images/ackley_true.pdf')
