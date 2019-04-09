#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/seq_des.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 03.31.2019

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

## Sequential design of an acquisition function using a deep kernel.
def seq_design(design, response, model, objective, seq_steps, y_mu, y_sig, explore_starts = 10, verbose = False):
    """
    Sequential Acquisition.
    """
    #########
    N = design.shape[0]
    P = design.shape[1]
    ## Initial model
    design_tf = tf.Variable(design)
    response_tf = tf.Variable(response.reshape([N,1]))
    Z = model(tf.cast(design_tf, tf.float32))

    # Initial fit to get amplitude (this is horribly inefficient but easiest for now)
    kernel = psd_kernels.ExponentiatedQuadratic(amplitude = np.array([1]).astype(np.float32), length_scale = np.array([0.1]).astype(np.float32))
    gp = tfd.GaussianProcess(kernel, Z, jitter = tf.cast(tf.Variable(nugget), tf.float32))

    ## Plug in estimate
    #K = tf.squeeze(gp.covariance())
    #tau_hat = tf.squeeze(tf.sqrt(tf.matmul(tf.transpose(response_tf), tf.linalg.solve(tf.cast(K, tf.float64), response_tf)) / float(N)))
    #kernel = psd_kernels.ExponentiatedQuadratic(amplitude = np.array([tau_hat]).astype(np.float32), length_scale = np.array([0.1]).astype(np.float32))
    #gp = tfd.GaussianProcess(kernel, Z, jitter = tf.cast(tf.Variable(nugget), tf.float32))

    #########
    ## Find the next point
    explored = []
    for it in range(seq_steps):
        # Start with 2 inits: one maximizing variance, and one at the previous optimum
        # Prev optim
        init_x = design_tf.numpy()[np.argmin(response),:] 
        #print(init_x)
        #print(spy_neur_nei(init_x))
        #print(spy_neur_nei_grad(init_x))

        optret = minimize(spy_neur_nei, init_x, bounds = [(0,1) for _ in range(P)], method = 'L-BFGS-B',\
                jac = spy_neur_nei_grad, args = (model, gp, response_tf))
        optret
        exploit_val = optret.fun
        exploit_x = optret.x

        # Max var, initing that randomly, then into EI
        explore_vals = np.empty(explore_starts)
        explore_xs = []
        for eit in range(explore_starts):
            init_x = np.random.uniform(size=P)
            #print(init_x)
            #print(spy_nvar(init_x))
            #print(spy_nvar_grad(init_x))

            optret = minimize(spy_nvar, init_x, bounds = [(0,1) for _ in range(P)], method = 'L-BFGS-B',\
                    jac = spy_nvar_grad, args = (model, gp, response_tf))
            optret
            var_init = optret.x

            #print(spy_neur_nei(var_init))
            #print(spy_neur_nei_grad(var_init))

            optret = minimize(spy_neur_nei, var_init, bounds = [(0,1) for _ in range(P)], method = 'L-BFGS-B',\
                    jac = spy_neur_nei_grad, args = (model, gp, response_tf))
            explore_vals[eit] = optret.fun
            explore_xs.append(optret.x)
        explore_val = min(explore_vals)
        explore_x = explore_xs[np.argmax(explore_vals)]

        if explore_val < exploit_val:
            if verbose:
                print("Exploring...")
            explored.append(True)
            new_x = explore_x
        else:
            if verbose:
                print("Exploiting..")
            explored.append(False)
            new_x = exploit_x
        new_y = (objective(new_x) - y_mu) / y_sig

        N += 1
        design = np.vstack([design, new_x])
        design_tf = tf.Variable(design)
        response = np.append(response, new_y)
        response_tf = tf.Variable(response.reshape([N,1]))
        Z = model(tf.cast(design_tf, tf.float32))

        # Plug in estimate
        #K = tf.squeeze(gp.covariance())
        #tau_hat = tf.squeeze(tf.sqrt(tf.matmul(tf.transpose(response_tf), tf.linalg.solve(tf.cast(K, tf.float64), response_tf)) / float(N)))
        #kernel = psd_kernels.ExponentiatedQuadratic(amplitude = np.array([tau_hat]).astype(np.float32), length_scale = np.array([0.1]).astype(np.float32))
        #gp = tfd.GaussianProcess(kernel, Z, jitter = tf.cast(tf.Variable(nugget), tf.float32))
        kernel = psd_kernels.ExponentiatedQuadratic(amplitude = np.array([1]).astype(np.float32), length_scale = np.array([0.1]).astype(np.float32))
        gp = tfd.GaussianProcess(kernel, Z, jitter = tf.cast(tf.Variable(nugget), tf.float32))

    return (design, response, explored)
