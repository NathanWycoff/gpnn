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
import time

start_time = time.time()

tf.random.set_random_seed(12345)
np.random.seed(12345)

GLOBAL_NUGGET = 1e-6

N_init = 20
P = 20
L = 2
H = 100
R = 2
seq_steps = 3
outer_bois = 2
saveit = False

save_title = ''.join([str(x) for x in ['./data/emb_quad_bakeoff_', L, '_', H, '_', R, '.npz']])

# Record some things
nei_min_obj = np.empty([seq_steps, outer_bois]) + np.nan
nme_min_obj = np.empty([seq_steps, outer_bois]) + np.nan
van_min_obj = np.empty([seq_steps, outer_bois]) + np.nan

nei_weight_diff = np.empty([seq_steps, outer_bois]) + np.nan
nme_weight_diff = np.empty([seq_steps, outer_bois]) + np.nan

nei_wup_iters = np.empty([seq_steps, outer_bois]) + np.nan
nme_wup_iters = np.empty([seq_steps, outer_bois]) + np.nan

# The initial model for the neural boys, give a plain-jane model to the vanilla boi.
for ob in range(outer_bois):
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

    # Define true function
    true_model = random_nn(P,L,H,R, act = tf.nn.tanh)
    design = neural_maxent(N_init, P, L, H, R, net_weights = true_model.get_weights())['design']
    true_extent = get_extent(design, true_model)
    def bb_obj(x):
        z = true_model(x.reshape([1,P])).numpy().flatten()
        for r in range(R):
            z[r] = (z[r] - true_extent[r][0]) / (true_extent[r][1] - true_extent[r][0])
        return(sum(np.square(z)))

    # People get to observe responses
    nei_response = np.apply_along_axis(bb_obj, 1, nei_design)
    nme_response = np.apply_along_axis(bb_obj, 1, nme_design)
    van_response = np.apply_along_axis(bb_obj, 1, van_design)

    #TODO: Do we need to help with the mean adjustment for the other models too?
    van_model.set_weights([van_model.get_weights()[0], np.repeat(np.mean(van_response), P)])

    # Update model after initial design for the neural guys.
    nei_model, _ = update_weights(nei_design, nei_response, nei_model, l2_coef = 0)
    nme_model, _ = update_weights(nei_design, nme_response, nme_model, l2_coef = 0)

    for si in range(seq_steps):
        # Sample a next point; append it to our design
        # NEI
        next_point = neural_maxent(1, P, L, H, R, X_prev = nme_design, net_weights = nme_model.get_weights())['design']
        next_y = bb_obj(next_point)
        nme_design = np.vstack([nme_design, next_point])
        nme_response = np.append(nme_response, next_y)
        # NME
        aug_design = seq_design(nei_design, nei_response, nei_model, bb_obj, seq_steps = 1, y_mu = 0, y_sig = 1, explore_starts = 10, verbose = False)[0]
        next_point = aug_design[-1,:]
        next_y = bb_obj(next_point)
        nei_design = np.vstack([nei_design, next_point])
        nei_response = np.append(nei_response, next_y)
        # Van
        aug_design = seq_design(van_design, van_response, van_model, bb_obj, seq_steps = 1, y_mu = 0, y_sig = 1, explore_starts = 10, verbose = False)[0]
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
        van_model, _ = update_weights(van_design, van_response, van_model, l2_coef = 0, diag = True)

        # Record some facts.
        nei_weight_diff[si,ob] = np.sqrt(sum(np.square(nei_weights_before - weights_2_vec(nei_model.get_weights()))))
        nme_weight_diff[si,ob] = np.sqrt(sum(np.square(nme_weights_before - weights_2_vec(nme_model.get_weights()))))
        nei_wup_iters[si, ob] = nei_nit
        nme_wup_iters[si ,ob] = nme_nit
        nei_min_obj[si, ob] = min(nei_response)
        nme_min_obj[si, ob] = min(nme_response)
        van_min_obj[si, ob] = min(van_response)

    print("iter took %s seconds"%(time.time() - start_time))
    if saveit:
        np.savez(save_title, nei_min_obj = nei_min_obj, nme_min_obj = nme_min_obj, van_min_obj = van_min_obj, \
                nei_weight_diff = nei_weight_diff, nme_weight_diff = nme_weight_diff, nei_wup_iters = nei_wup_iters, \
                nme_wup_iters = nme_wup_iters)
