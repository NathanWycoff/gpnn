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
import matplotlib.pyplot as plt
plt.ion()
from scipy.special import expit, logit
from scipy.spatial import distance_matrix
exec(open("python/hilbert_curve.py").read())
exec(open("python/ackley.py").read())
exec(open("python/neural_maxent.py").read())

## Sequential design of an acquisition function using a deep kernel.
## A toy design: start with some high D function, map it to R D using a MLP, 
## then do sequential design pretending like we know the true mapping. 
## We should demolish methods which do not avail themselves of this info.
N = 100
P = 3
L = 2
H = 10
R = 2
seq_steps = 10
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

#########
## Get a design
def test_objective(x):
    """
    An ackley defined on a low D space.
    """
    xs = x.reshape([1,x.shape[0]])
    z = model(tf.cast(xs, tf.float32)).numpy().reshape(R)
    return(ackley(z))

# Entropy max initial design
design = neural_maxent(N ,P, L, H, R, net_weights = model.get_weights())['design']
response_us = np.apply_along_axis(test_objective, 1, design)
response = (response_us - np.mean(response_us)) / np.std(response_us)

#########
## Initial model
design_tf = tf.Variable(design)
response_tf = tf.Variable(response.reshape([N,1]))
Z = model(tf.cast(design_tf, tf.float32))

# Initial fit to get amplitude (this is horribly inefficient but easiest for now)
kernel = psd_kernels.ExponentiatedQuadratic(amplitude = np.array([1]).astype(np.float32), length_scale = np.array([0.1]).astype(np.float32))
gp = tfd.GaussianProcess(kernel, Z, jitter = tf.cast(tf.Variable(1E-6), tf.float32))

# Plug in estimate
K = tf.squeeze(gp.covariance())
tau_hat = tf.squeeze(tf.sqrt(tf.matmul(tf.transpose(response_tf), tf.linalg.solve(tf.cast(K, tf.float64), response_tf)) / float(N)))
kernel = psd_kernels.ExponentiatedQuadratic(amplitude = np.array([tau_hat]).astype(np.float32), length_scale = np.array([0.1]).astype(np.float32))
gp = tfd.GaussianProcess(kernel, Z, jitter = tf.cast(tf.Variable(1E-6), tf.float32))

#########
## Find the next point
# Start with 2 inits: one maximizing variance, and one at the previous optimum
# Prev optim
init_x = design_tf.numpy()[np.argmin(response),:] 
print(init_x)
print(spy_nei(init_x))
print(spy_nei_grad(init_x))

optret = minimize(spy_nei, init_x, bounds = [(0,1) for _ in range(P)], method = 'L-BFGS-B',\
        jac = spy_nei_grad)
optret
exploit_val = optret.fun
exploit_x = optret.x

# Max var, initing that randomly, then into EI
init_x = np.random.uniform(size=P)
print(init_x)
print(spy_nvar(init_x))
print(spy_nvar_grad(init_x))

optret = minimize(spy_nvar, init_x, bounds = [(0,1) for _ in range(P)], method = 'L-BFGS-B',\
        jac = spy_nvar_grad)
optret
var_init = optret.x

print(spy_nei(var_init))
print(spy_nei_grad(var_init))

optret = minimize(spy_nei, var_init, bounds = [(0,1) for _ in range(P)], method = 'L-BFGS-B',\
        jac = spy_nei_grad)
optret
explore_val = optret.fun
explore_x = optret.x

if explore_val < exploit_val:
    print("Exploring...")
    new_x = explore_x
else:
    print("Expoiting..")
    new_x = exploit_x
