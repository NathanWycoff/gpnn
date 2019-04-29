#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/ackley_bakeoff.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.21.2019

## Do sequential design on the Ackley function, comparing a couple methods.
#TODO: Nugget
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

tf.random.set_random_seed(12345)
np.random.seed(12345)

N_init = 20
P = 20
L = 2
H = 100
R = 1
seq_steps = 15*P

# The initial model for the neural boys, give a plain-jane model to the vanilla boi.
init_model = random_nn(P,L,H,R, act = tf.nn.tanh)

nei_model = tf.keras.models.clone_model(init_model)
nme_model = tf.keras.models.clone_model(init_model)
nei_model.set_weights(init_model.get_weights())
nme_model.set_weights(init_model.get_weights())

van_model = random_nn(P,0,None,P)
van_model.set_weights([0.1*np.eye(P), np.zeros(P)])

# Get a Maximum Entropy design for everybody
#init_w = used_model.get_weights()
nei_design = neural_maxent(N_init, P, L, H, R, net_weights = nei_model.get_weights())['design']
nme_design = neural_maxent(N_init, P, L, H, R, net_weights = nme_model.get_weights())['design']
van_design = neural_maxent(N_init, P, 0, None, P, net_weights = van_model.get_weights())['design']

# Get mean,var for this function in this dim
design = np.random.uniform(size=[10000,P])
response_us = np.apply_along_axis(lambda x: ackley(x.flatten()), 1, design)
y_mu = np.mean(response_us)
y_sig = np.std(response_us)
# Off center it in a few dimensions
reseed = np.random.randint(1e5)
np.random.seed(123)
oc = np.random.uniform(-0.2,0.2, size = [P])
np.random.seed(reseed)
bb_obj = lambda x: (ackley(x.flatten() + oc) - y_mu) / y_sig

# People get to observe responses
nei_response = np.apply_along_axis(bb_obj, 1, nei_design)
nme_response = np.apply_along_axis(bb_obj, 1, nme_design)
van_response = np.apply_along_axis(bb_obj, 1, van_design)

# Update model after initial design for the neural guys.
nei_model, _ = update_weights(nei_design, nei_response, nei_model, l2_coef = 0)
nme_model, _ = update_weights(nei_design, nme_response, nme_model, l2_coef = 0)
#TODO: Let the van_gp update its lengthscales.

# Record some things
nei_min_obj = np.empty(seq_steps)
nme_min_obj = np.empty(seq_steps)
van_min_obj = np.empty(seq_steps)
nei_min_obj[:] = np.nan
nme_min_obj[:] = np.nan
van_min_obj[:] = np.nan

nei_weight_diff = np.zeros(seq_steps) + np.nan
nme_weight_diff = np.zeros(seq_steps) + np.nan

nei_wup_iters = np.zeros(seq_steps) + np.nan
nme_wup_iters = np.zeros(seq_steps) + np.nan

for si in range(seq_steps):
    # Sample a next point; append it to our design
    # NEI
    next_point = neural_maxent(1, P, L, H, R, X_prev = nei_design, net_weights = nei_model.get_weights())['design']
    next_y = bb_obj(next_point)
    nei_design = np.vstack([nei_design, next_point])
    nei_response = np.append(nei_response, next_y)
    # NME
    aug_design = seq_design(nme_design, nme_response, nme_model, bb_obj, seq_steps = 1, y_mu = 0, y_sig = 1, explore_starts = 10, verbose = True)[0]
    next_point = aug_design[-1,:]
    next_y = bb_obj(next_point)
    nme_design = np.vstack([nme_design, next_point])
    nme_response = np.append(nme_response, next_y)
    # Van
    aug_design = seq_design(van_design, van_response, van_model, bb_obj, seq_steps = 1, y_mu = 0, y_sig = 1, explore_starts = 10, verbose = True)[0]
    next_point = aug_design[-1,:]
    next_y = bb_obj(next_point)
    van_design = np.vstack([van_design, next_point])
    van_response = np.append(van_response, next_y)

    #TODO: Does this converge? Do, eventually, we learn a W and stop learning mutch upon a new point being revealed?
    # Record a prior weights
    nei_weights_before = weights_2_vec(nei_model.get_weights())
    nme_weights_before = weights_2_vec(nme_model.get_weights())
    # Update our neural net weights.
    nei_model, nei_nit = update_weights(nei_design, nei_response, nei_model, l2_coef = 0)
    nme_model, nme_nit = update_weights(nme_design, nme_response, nme_model, l2_coef = 0)

    # Record some facts.
    nei_weight_diff[si] = np.sqrt(sum(np.square(nei_weights_before - weights_2_vec(nei_model.get_weights()))))
    nme_weight_diff[si] = np.sqrt(sum(np.square(nme_weights_before - weights_2_vec(nme_model.get_weights()))))
    nei_wup_iters[si] = nei_nit
    nme_wup_iters[si] = nme_nit
    nei_min_obj[si] = min(nei_response)
    nme_min_obj[si] = min(nme_response)
    van_min_obj[si] = min(van_response)

title = ''.join([str(x) for x in ['./data/ackley_bakeoff_', L, '_', H, '_', R, '.npz']])
np.savez(title, nei_min_obj = nei_min_obj, nme_min_obj = nme_min_obj, van_min_obj = van_min_obj, \
        nei_weight_diff = nei_weight_diff, nme_weight_diff = nme_weight_diff, nei_wup_iters = nei_wup_iters, \
        nme_wup_iters = nme_wup_iters)
