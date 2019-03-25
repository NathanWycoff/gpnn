#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/take3.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 03.23.2019

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
tf.enable_eager_execution()
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.ion()
from scipy.special import expit

tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels

N = 10
P = 2
H = 100
R = 1
learn_rate = 0.01
iters = 1000
kernel = psd_kernels.ExponentiatedQuadratic()

def scipy_cost(x):
    X.assign(np.array(x).reshape([N,P]))
    return loss(X).numpy()

def scipy_grad(x):
    X.assign(np.array(x).reshape([N,P]))
    with tf.GradientTape() as t:
        l = loss(X)
    return (t.gradient(l, X).numpy()).flatten()

#init_design = np.random.uniform(size=[N,P])
init_design = np.random.normal(0,0.001, size = [N,P])
X = tf.Variable(init_design)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
#    tf.keras.layers.Dense(H, activation=tf.nn.sigmoid, input_shape=(P,)),
    tf.keras.layers.Dense(R, input_shape = (P,))
    ])

def loss(X):
    Z = model(tf.math.sigmoid(X))

    gp = tfd.GaussianProcess(kernel, Z, jitter = 1E-10)

    nldetK = -tf.linalg.logdet(gp.covariance())
    return nldetK

loss(X)

### Optimize with Steepest Descent
#for it in range(iters):
#    with tf.GradientTape() as t:
#        l = loss(X)
#
#    dX = t.gradient(l, X)
#
#    X.assign(tf.clip_by_value(X - learn_rate * dX, 0, 1))
#    print(l)
#
#fig = plt.figure()
#
#X_sol = X.numpy()
#plt.subplot(2,2,1)
#plt.scatter(X_sol[:,0], X_sol[:,1])
#plt.title(l)
#
#plt.subplot(2,2,2)
#plt.scatter(init_design[:,0], init_design[:,1])
#plt.title(scipy_cost(init_design))
#
#Z_sol = model(X_sol)
#plt.subplot(2,2,3)
##plt.scatter(Z_sol[:,0], Z_sol[:,1])
#plt.scatter([0 for _ in range(N)], Z_sol[:])
#plt.title("In Latent Space")
#
#plt.show()

### With SCIPY BFGS
#optret = minimize(scipy_cost, init_design, bounds = [(0,1) for _ in range(N*P)], method = 'L-BFGS-B',\
#        jac = scipy_grad, tol = 0)
optret = minimize(scipy_cost, init_design, method = 'BFGS',\
        jac = scipy_grad, tol = 0)

print(optret.success)
print(optret.fun)
X_sol = optret.x.reshape([N,P])

fig = plt.figure()

plt.subplot(2,2,1)
plt.scatter(X_sol[:,0], X_sol[:,1])
plt.title(optret.fun)

plt.subplot(2,2,2)
plt.scatter(init_design[:,0], init_design[:,1])
plt.title(scipy_cost(init_design))

Z_sol = model(expit(X_sol))
plt.subplot(2,2,3)
#plt.scatter(Z_sol[:,0], Z_sol[:,1])
if R == 1:
    plt.scatter([0 for _ in range(N)], Z_sol[:])
elif R == 2:
    plt.scatter(Z_sol[:,0], Z_sol[:,1])
plt.title("In Latent Space")


plt.show()
