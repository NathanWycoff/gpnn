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

## Sequential design of an acquisition function using a deep kernel.
## A toy design: start with some high D function, map it to R D using a MLP, 
## then do sequential design pretending like we know the true mapping. 
## We should demolish methods which do not avail themselves of this info.
N = 100
P = 10
L = 2
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


def test_objective(x):
    """
    An ackley defined on a low D space.
    """
    xs = x.reshape([1,x.shape[0]])
    z = model(tf.cast(xs, tf.float32)).numpy().reshape(R)
    return(ackley(z))

# Entropy max initial design
design = neural_maxent(N ,P, L, H, R, net_weights = model.get_weights())['design']
response = np.apply_along_axis(test_objective, 1, design)

# Define likelihood wrt weights
def weights_loss(weights):
    #### Compute low D subspace.
    model.set_weights(weights)
    Z = model(tf.cast(design, np.float32))

    # Get the entropy of the design
    gp = tfd.GaussianProcess(kernel, Z, jitter = 1E-6)
    nll = -gp.log_prob(response)

    return nll

def weights_2_vec(weights):
    return(np.concatenate([wx.flatten() for wx in weights]))

def vec_2_weights(vec):
    nlayers = L+1
    weights = []
    used_neurons = 0
    prev_shape = P
    for l in range(nlayers):
        # Add connection weights:
        curr_size = model.layers[l].output_shape[1] * prev_shape
        weights.append(vec[used_neurons:(used_neurons+curr_size)].reshape([ prev_shape, model.layers[l].output_shape[1]]))
        used_neurons += curr_size

        # Add biases:
        curr_size = model.layers[l].output_shape[1] 
        vec[used_neurons:(used_neurons+curr_size)]
        weights.append(vec[used_neurons:(used_neurons+curr_size)])
        used_neurons += curr_size

        prev_shape = model.layers[l].output_shape[1]

    return(weights)

def spy_weights_cost(x):
    weights = vec_2_weights(x)
    weights_loss(weights).numpy()

def spy_weights_grad(x):
    weights = vec_2_weights(x)
    weights_loss(weights).numpy()

def spy_maxent_grad(x):
    X.assign(np.array(x).reshape([N,P]))
    with tf.GradientTape() as t:
        l = maxent_loss(X)
    return (t.gradient(l, X).numpy()).flatten()
