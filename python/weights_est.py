#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/weights_est.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.08.2019

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

## Sequential design of an acquisition function using a deep kernel.
## A toy design: start with some high D function, map it to R D using a MLP, 
## then do sequential design pretending like we know the true mapping. 
## We should demolish methods which do not avail themselves of this info.
N_init = 1000
P = 3
L = 1
H = 10
R = 2
seq_steps = 30
explore_starts = 10
# TODO: These next two dials are pretty fragile.
minalldist = 1e-5
nugget = 1E-5
scalealldist = 1500

## Model we're using
tf.random.set_random_seed(1234)
np.random.seed(1234)
act = tf.nn.tanh
used_model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(H, activation=act, input_shape=(P,)) if i == 0 else tf.keras.layers.Dense(H, activation=act) for i in range(L)] + 
    [tf.keras.layers.Dense(R)
    ])
used_model.build(input_shape=[P])
used_model.summary()

# The data-generating model
act = tf.nn.tanh
true_model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(H, activation=act, input_shape=(P,)) if i == 0 else tf.keras.layers.Dense(H, activation=act) for i in range(L)] + 
    [tf.keras.layers.Dense(R)
    ])
true_model.build(input_shape=[P])
true_model.summary()
#true_model = used_model

## Get the extent of the design with many points.
viz_design = neural_maxent(100 ,P, L, H, R, net_weights = used_model.get_weights())['design']
viz_Z = used_model(tf.cast(viz_design, tf.float32)).numpy()
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
    z = true_model(tf.cast(xs, tf.float32)).numpy().reshape(R)
    # Reshape according to the extent of the low D points.
    return(myackley(z))

# Entropy max initial design
N = N_init
design = neural_maxent(N ,P, L, H, R, net_weights = used_model.get_weights())['design']
design_tf = tf.Variable(design)
response_us = np.apply_along_axis(test_objective, 1, design)
y_mu = np.mean(response_us)
y_sig = np.std(response_us)
response = (response_us - y_mu) / y_sig

## Sample many points to make a response surface in the "wrong"/used model
B = 100000
X_samp = np.random.uniform(size=[B,P])
Zu_samp = used_model(tf.cast(X_samp, tf.float32))
y_samp = np.apply_along_axis(test_objective, 1, X_samp)
pred = KNeighborsRegressor(n_neighbors = 5)
pred.fit(Zu_samp, y_samp)
delta = 0.025
x = np.arange(extent[0], extent[1], delta)
y = np.arange(extent[2], extent[3], delta)
X, Y = np.meshgrid(x, y)
toplot = np.empty(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        toplot[i,j] = pred.predict(np.array([X[i,j],Y[i,j]]).reshape(1,-1))
Z_sol = used_model(tf.cast(design_tf, tf.float32)).numpy()

fig, ax = plt.subplots()
im = ax.imshow(toplot, interpolation='bilinear', origin='lower',
                cmap=cm.gray, extent=extent)
plt.scatter(Z_sol[:,0], Z_sol[:,1], c = response)
plt.autumn()
plt.show()
plt.savefig('images/ackley_before.pdf')

## Estimate the weights!
est_model = update_weights(design, response, used_model, l2_coef = 0.5)

## Get the extent of the design with many points.
viz_design = neural_maxent(100 ,P, L, H, R, net_weights = est_model.get_weights())['design']
viz_Z = est_model(tf.cast(viz_design, tf.float32)).numpy()
extent = [min(viz_Z[:,0]), max(viz_Z[:,0]), min(viz_Z[:,1]), max(viz_Z[:,1])]

## Sample many points to make a response surface in the "wrong"/used model
B = 100000
X_samp = np.random.uniform(size=[B,P])
Zu_samp = est_model(tf.cast(X_samp, tf.float32))
y_samp = np.apply_along_axis(test_objective, 1, X_samp)
pred = KNeighborsRegressor(n_neighbors = 5)
pred.fit(Zu_samp, y_samp)
delta = 0.025
x = np.arange(extent[0], extent[1], delta)
y = np.arange(extent[2], extent[3], delta)
X, Y = np.meshgrid(x, y)
toplot = np.empty(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        toplot[i,j] = pred.predict(np.array([X[i,j],Y[i,j]]).reshape(1,-1))
Z_sol = est_model(tf.cast(design_tf, tf.float32)).numpy()

fig, ax = plt.subplots()
im = ax.imshow(toplot, interpolation='bilinear', origin='lower',
                cmap=cm.gray, extent=extent)
plt.scatter(Z_sol[:,0], Z_sol[:,1], c = response)
plt.autumn()
plt.show()
plt.savefig('images/ackley_after.pdf')
