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
N = N_init
design = neural_maxent(N ,P, L, H, R, net_weights = model.get_weights())['design']
response = np.apply_along_axis(ackley_obj, 1, design)

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

# Define likelihood wrt weights
def weights_loss(weights, model, response):
    model.set_weights(weights)
    Z = model(tf.cast(design, np.float32))

    # Get the entropy of the design
    gp = tfd.GaussianProcess(kernel, Z, jitter = 1E-6)
    nll = -gp.log_prob(response)

    return nll


def spy_weights_cost(w, model, response):
    weights = vec_2_weights(w)
    return float(weights_loss(weights, model, response).numpy())

def spy_weights_grad(w, model, response):
    weights = vec_2_weights(w)
    with tf.GradientTape() as t:
        nll = weights_loss(weights, model, response)
    dweights = t.gradient(nll, model.trainable_weights)
    dweightsnp = [wi.numpy().astype(np.float64) for wi in dweights]

    return weights_2_vec(dweightsnp)

# Do an optimization boy.
#TODO: reliance on globals in opt
init_w = weights_2_vec(model.get_weights()).astype(np.float64)
optret = minimize(spy_weights_cost, init_w, method = 'L-BFGS-B',\
        jac = spy_weights_grad, args = (model, response))

model.set_weights(vec_2_weights(init_w))
model(tf.cast(design, np.float32))
model.set_weights(vec_2_weights(optret.x))
model(tf.cast(design, np.float32))
