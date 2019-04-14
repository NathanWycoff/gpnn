#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/ackley_radius.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.14.2019

## Can the gpnn learn that only radius from the center matters?
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
from sklearn.metrics import r2_score
exec(open("python/hilbert_curve.py").read())
exec(open("python/neural_maxent.py").read())
exec(open("python/seq_des.py").read())
exec(open("python/opt_lib.py").read())
exec(open("python/misc.py").read())
exec(open("python/test_funcs.py").read())

tf.random.set_random_seed(1234)
np.random.seed(1234)

N_init = 1000
P = 3
L = 2
H = 10
R = 1

# Two similarly shaped random nets, one is the one we init on, one the one we use.
used_model = random_nn(P,L,H,R, act = tf.nn.tanh)

init_w = used_model.get_weights()
design = neural_maxent(N_init ,P, L, H, R, net_weights = used_model.get_weights())['design']

bb_obj = ackley

response_us = np.apply_along_axis(bb_obj, 1, design)
y_mu = np.mean(response_us)
y_sig = np.std(response_us)
response = (response_us - y_mu) / y_sig

est_model = update_weights(design, response, used_model, l2_coef = 0)

# Does the low D viz correspond to a radius?
init_preds = used_model.predict(design)
preds = est_model.predict(design)
rads = np.apply_along_axis(np.linalg.norm, 1, design)
fig = plt.figure()

plt.subplot(1,2,1)
plt.scatter(init_preds, rads)
plt.xlabel('NN output')
plt.ylabel('Norm of point')
r2 = pow(np.corrcoef(init_preds.flatten(), rads)[0,1], 2)
plt.title('Initial (r2 = %s)'%round(r2, 4))

plt.subplot(1,2,2)
plt.scatter(preds, rads)
plt.xlabel('NN output')
plt.ylabel('Norm of point')
r2 = pow(np.corrcoef(preds.flatten(), rads)[0,1], 2)
plt.title('Learned (r2 = %s)'%round(r2, 4))
[1,1]
plt.savefig('images/temp.pdf')
