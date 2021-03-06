#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/known_weights.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.01.2019

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
exec(open("python/hilbert_curve.py").read())
exec(open("python/ackley.py").read())
exec(open("python/neural_maxent.py").read())
exec(open("python/seq_des.py").read())
exec(open("python/opt_lib.py").read())

## Sequential design of an acquisition function using a deep kernel.
## A toy design: start with some high D function, map it to R D using a MLP, 
## then do sequential design pretending like we know the true mapping. 
## We should demolish methods which do not avail themselves of this info.
N_init = 10
P = 5
L = 1
H = 10
R = 2
seq_steps = 10
explore_starts = 10
# TODO: These next two dials are pretty fragile.
minalldist = 1e-5
nugget = 1E-5
scalealldist = 1500

tf.random.set_random_seed(123)
np.random.seed(123)
act = tf.nn.tanh
model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(H, activation=act, input_shape=(P,)) if i == 0 else tf.keras.layers.Dense(H, activation=act) for i in range(L)] + 
    [tf.keras.layers.Dense(R)
    ])
model.build(input_shape=[P])
model.summary()

## Get the extent of the design with many points.
viz_design = neural_maxent(100 ,P, L, H, R, net_weights = model.get_weights())['design']
viz_Z = model(tf.cast(viz_design, tf.float32)).numpy()
extent = [min(viz_Z[:,0]), max(viz_Z[:,0]), min(viz_Z[:,1]), max(viz_Z[:,1])]

#########
def myackley(z):
    assert R == 2
    z[0] = (z[0] - extent[0]) / (extent[1] - extent[0])
    z[1] = (z[1] - extent[2]) / (extent[3] - extent[2])
    return(ackley(z))

## Get a design
def test_objective(x):
    """
    An ackley defined on a low D space.
    """
    xs = x.reshape([1,x.shape[0]])
    z = model(tf.cast(xs, tf.float32)).numpy().reshape(R)
    # Reshape according to the extent of the low D points.
    return(myackley(z))

# Entropy max initial design
N = N_init
design = neural_maxent(N ,P, L, H, R, net_weights = model.get_weights())['design']
response_us = np.apply_along_axis(test_objective, 1, design)
y_mu = np.mean(response_us)
y_sig = np.std(response_us)
response = (response_us - y_mu) / y_sig

design, response, explored = seq_design(design = design, response = response, model = model, objective = test_objective, seq_steps = seq_steps, explore_starts = explore_starts, verbose = True)
design_tf = tf.Variable(design)

## Contour plot
delta = 0.025
x = np.arange(extent[0], extent[1], delta)
y = np.arange(extent[2], extent[3], delta)
X, Y = np.meshgrid(x, y)
toplot = np.empty(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        toplot[i,j] = myackley(np.array([X[i,j], Y[i,j]]))

Z_sol = model(tf.cast(design_tf, tf.float32)).numpy()
fig, ax = plt.subplots()

im = ax.imshow(toplot, interpolation='bilinear', origin='lower',
                cmap=cm.gray, extent=extent)
plt.scatter(Z_sol[:N_init,0], Z_sol[:N_init,1])
#plt.scatter(Z_sol[N_init:,0], Z_sol[N_init:,1])
for i in range(N_init, N_init+seq_steps):
    ind = i - N_init + 1
    if explored[ind-1]:
        ax.text(Z_sol[i,0], Z_sol[i,1], str(ind), color='orange', fontweight='bold')
    else:
        ax.text(Z_sol[i,0], Z_sol[i,1], str(ind), color='red', fontweight='bold')

plt.show()

plt.savefig('images/ackley_known.pdf')
