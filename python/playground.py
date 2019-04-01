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
from scipy.special import expit, logit
from scipy.spatial import distance_matrix
exec(open("python/hilbert_curve.py").read())

tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels

N = 100
P = 3
L = 1
H = 10
R = 2
learn_rate = 0.01
iters = 1000
# TODO: These next two dials are pretty fragile.
minalldist = 1e-5
scalealldist = 1500
kernel = psd_kernels.ExponentiatedQuadratic(length_scale = np.array([0.1]).astype(np.float32))
in_sigspace = False

def scipy_cost(x):
    X.assign(np.array(x).reshape([N,P]))
    return loss(X).numpy()

def scipy_grad(x):
    X.assign(np.array(x).reshape([N,P]))
    with tf.GradientTape() as t:
        l = loss(X)
    return (t.gradient(l, X).numpy()).flatten()

init_design = hc_design(N,P)
if in_sigspace:
    init_design = logit((init_design+0.1)/1.1)
X = tf.Variable(init_design)

act = tf.nn.tanh
model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(H, activation=act, input_shape=(P,)) if i == 0 else tf.keras.layers.Dense(H, activation=act) for i in range(L)] + 
    [tf.keras.layers.Dense(R)
    ])
model.build(input_shape=[P])
model.summary()

# Set to identity if desired:
#model.set_weights([np.eye(2), np.zeros(2)])

def bump(x, lb, ub, scale = 10):
    """
    A smooth function which will be large between lb and ub, and zero elsewhere.

    :param x: The function argument.
    :param lb, ub: Function will be nonzero only within lb, ub.
    :param scale: Max value of function.
    """
    xn11 = 2 * (x-lb) / (ub - lb) - 1
    return scale * np.e * tf.exp(-1/(1-tf.square(tf.clip_by_value(xn11,-1,1))))

def loss(X):
    #### Compute low D subspace.
    if in_sigspace:
        Z = model(tf.cast(tf.math.sigmoid(X), tf.float32))
    else:
        Z = model(tf.cast(X, tf.float32))

    #### Small Distance penalty.
    r = tf.reduce_sum(X*X, 1)

    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(X, tf.transpose(X)) + tf.transpose(r)

    Da = D + tf.cast(np.power(P, 2) * tf.eye(N), np.float64)
    mindist = tf.reduce_min(Da)
    distpen = tf.cast(bump(mindist, -minalldist, minalldist, scalealldist), tf.float32)

    # Detect if near boundary
    #bump(X, )
    
    # Get the entropy of the design
    gp = tfd.GaussianProcess(kernel, Z, jitter = 1E-6)
    nldetK = -tf.linalg.logdet(gp.covariance())

    return nldetK + distpen

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
if in_sigspace:
    optret = minimize(scipy_cost, init_design, method = 'BFGS',\
            jac = scipy_grad, tol = 0)
    ides = expit(init_design)
else:
    optret = minimize(scipy_cost, init_design, bounds = [(0,1) for _ in range(N*P)], method = 'L-BFGS-B',\
            jac = scipy_grad, options = {'ftol' : 0})
    ides = init_design

print(optret.success)
print(optret.fun)
if in_sigspace:
    X_sol = expit(optret.x.reshape([N,P]))
else:
    X_sol = optret.x.reshape([N,P])

fig = plt.figure(figsize = [8,8])

plt.subplot(2,2,1)
plt.scatter(ides[:,0], ides[:,1])
plt.title("Initial Design (cost %s)"%np.around(scipy_cost(ides), decimals = 2))

plt.subplot(2,2,2)
plt.scatter(X_sol[:,0], X_sol[:,1])
plt.title("Final Design (cost %s)"%np.around(optret.fun))

Z_sol = model(tf.cast(ides, tf.float32))
plt.subplot(2,2,3)
if R == 1:
    plt.scatter([0 for _ in range(N)], Z_sol[:])
elif R == 2:
    plt.scatter(Z_sol[:,0], Z_sol[:,1])
plt.title("Initial Latent Representation")

Z_sol = model(tf.cast(X_sol, tf.float32))
plt.subplot(2,2,4)
if R == 1:
    plt.scatter([0 for _ in range(N)], Z_sol[:])
elif R == 2:
    plt.scatter(Z_sol[:,0], Z_sol[:,1])
plt.title("Final Latent Representation")

plt.show()

plt.savefig('temp.pdf')

Dmat = distance_matrix(X_sol, X_sol)
np.min(Dmat[np.triu_indices(N, k = 1)])
