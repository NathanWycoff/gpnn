#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/neural_maxent.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 03.31.2019

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
import warnings
exec(open("python/hilbert_curve.py").read())

#TODO: Is this really the right place for this?
tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels

def bump(x, lb, ub, scale = 10):
    """
    A smooth function which will be large between lb and ub, and zero elsewhere.

    :param x: The function argument.
    :param lb, ub: Function will be nonzero only within lb, ub.
    :param scale: Max value of function.
    """
    xn11 = 2 * (x-lb) / (ub - lb) - 1
    return scale * np.e * tf.exp(-1/(1-tf.square(tf.clip_by_value(xn11,-1,1))))


#TODO: Make this a class to be more pythonic?
#TODO: Should pass the model object instead of P,L,H,R and net_weights.
#TODO: Should issue a warning if few iterations occur.
def neural_maxent(N, P, L, H, R, X_prev = None, minalldist = 1e-5, scalealldist = 1500, net_weights = None, act = tf.nn.tanh, verbose = False):
    """
    Create a neural maxentropy design with N points in P dimensions.

    :param N: The number of additional design points
    :param P: The dimension of the input space.
    :param L: The number of layers in the neural net.
    :param H: Width of hidden layers (scalar).
    :param R: The dimension of the output space.
    :param X_prev: The existing design to add to (optional).
    :param net_weights: The weights to initialize the neural net, to be passed to the set_weights method of the neural net object.

    """

    if X_prev is not None:
        N_total = N + X_prev.shape[0]
    else:
        N_total = N

    def maxent_loss(X):
        #### Compute low D subspace.
        Z = model(X)

        #### Small Distance penalty.
        r = tf.reduce_sum(X*X, 1)

        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        D = r - 2*tf.matmul(X, tf.transpose(X)) + tf.transpose(r)

        Da = D + tf.cast(np.power(P, 2) * tf.eye(N_total), np.float64)
        mindist = tf.reduce_min(Da)
        distpen = bump(mindist, -minalldist, minalldist, scalealldist)

        # Detect if near boundary
        
        # Get the entropy of the design
        #TODO: This seems like a pretty good default, but can we confirm that it works generally?
        kernel = psd_kernels.ExponentiatedQuadratic(amplitude = np.array([1.0]), length_scale = np.array([0.1]))
        gp = tfd.GaussianProcess(kernel, Z, jitter = GLOBAL_NUGGET)
        nldetK = -tf.linalg.logdet(gp.covariance())

        return nldetK + distpen


    if X_prev is None:
        new_points = hc_design(N,P)
    else:
        #TODO: Smarter init when adding
        new_points = np.random.uniform(size=[N,P])

    if X_prev is not None:
        init_design = np.vstack([new_points, X_prev])
    else:
        init_design = new_points
    X = tf.Variable(init_design)
    
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(H, activation=act, input_shape=(P,), dtype = tf.float64) if i == 0 else tf.keras.layers.Dense(H, activation=act, dtype = tf.float64) for i in range(L)] + 
        [tf.keras.layers.Dense(R, dtype = tf.float64) if L > 0 else tf.keras.layers.Dense(R, input_shape = (P,), dtype = tf.float64)]
        )
    model.build(input_shape=[P])
    if net_weights is not None:
        model.set_weights(net_weights)

    ### With SCIPY BFGS
    def spy_maxent_cost(x):
        """
        x is the vector of inputs, while X is a tensorflow matrix of appropriate size.
        """
        if X_prev is not None:
            X.assign(np.vstack([np.array(x).reshape([N,P]), X_prev]))
        else:
            X.assign(np.array(x).reshape([N,P]))

        return maxent_loss(X).numpy()
    def spy_maxent_grad(x):
        #TODO: Maybe avoid calculating gradients that we don't need?
        if X_prev is not None:
            X.assign(np.vstack([np.array(x).reshape([N,P]), X_prev]))
        else:
            X.assign(np.array(x).reshape([N,P]))
        with tf.GradientTape() as t:
            l = maxent_loss(X)
        all_grads = (t.gradient(l, X).numpy()).flatten()
        return all_grads[:(N*P)]

    in_new_points = new_points.flatten()
    optret = minimize(spy_maxent_cost, new_points.flatten(), bounds = [(0,1) for _ in range(N*P)], method = 'L-BFGS-B',\
            jac = spy_maxent_grad)
    #TODO: Var not used, get rid?
    ides = init_design

    if optret.success:
        msg = "Successful Search Termination"
    else:
        msg = "Abnormal Search Termination"
    if verbose:
        print(''.join([msg, ". Final Entropy = %f"%(-optret.fun)]))
    X_sol = optret.x.reshape([N,P])

    if optret.nit <= 5:
        warnings.warn("Warning: neural_maxent only took %d iters in minimization."%optret.nit)

    return {'design' : X_sol, 'entropy' : -float(optret.fun),  'optret' : optret,
            'init_design' : init_design, 'init_entropy' : -spy_maxent_cost(in_new_points), 'nnet' : model}

#TODO: Do updates from the above down here.
def neural_maxent_box(N, P, L, H, R, minalldist = 1e-5, scalealldist = 1500, outer_bfgs_iters = 1000, \
        ftol = 2.220446049250313e-09, gtol = 1e-05):
    """
    Create a neural maxentropy design with N points in P dimensions.
    Optimize via R B Gramacy's suggestion of bounding box iteration.
    """

    def maxent_loss(X):
        #### Compute low D subspace.
        Z = model(X)

        #### Small Distance penalty.
        r = tf.reduce_sum(X*X, 1)

        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        D = r - 2*tf.matmul(X, tf.transpose(X)) + tf.transpose(r)

        Da = D + tf.cast(np.power(P, 2) * tf.eye(N), np.float64)
        mindist = tf.reduce_min(Da)
        distpen = bump(mindist, -minalldist, minalldist, scalealldist)

        # Detect if near boundary
        
        # Get the entropy of the design
        gp = tfd.GaussianProcess(kernel, Z, jitter = GLOBAL_NUGGET)
        nldetK = -tf.linalg.logdet(gp.covariance())

        return nldetK + distpen


    init_design = hc_design(N,P)
    X = tf.Variable(init_design)

    act = tf.nn.tanh
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(H, activation=act, input_shape=(P,), dtype = tf.float64) if i == 0 else tf.keras.layers.Dense(H, activation=act, dtype = tf.float64) for i in range(L)] + 
        [tf.keras.layers.Dense(R, dtype = tf.float64) if L > 0 else tf.keras.layers.Dense(R, input_shape = (P,), dtype = tf.float64)]
        )
    model.build(input_shape=[P])

    ### With SCIPY BFGS
    def spy_maxent_cost(x):
        """
        x is the vector of inputs, while X is a tensorflow matrix of appropriate size.
        """
        X.assign(np.array(x).reshape([N,P]))
        return maxent_loss(X).numpy()
    def spy_maxent_grad(x):
        X.assign(np.array(x).reshape([N,P]))
        with tf.GradientTape() as t:
            l = maxent_loss(X)
        return (t.gradient(l, X).numpy()).flatten()

    cur_sol = init_design
    fdiff = np.inf
    flast = np.inf
    gnorm = np.inf
    it = 0
    while it < outer_bfgs_iters and fdiff > ftol and gnorm > gtol:
        print(it)
        # Conduct the optimization
        optret = minimize(spy_maxent_cost, cur_sol, bounds = [(0,1) for _ in range(N*P)], method = 'L-BFGS-B',\
                jac = spy_maxent_grad, options = {'maxiter' : 1})
        cur_sol = optret.x.reshape([N,P])

        # Check for convergence in function values
        if it > 0:
            fdiff = (flast - optret.fun)/max([abs(flast), abs(optret.fun),1])
        else:
            fdiff = np.inf
        flast = optret.fun

        # Check for convergence in gradient infinity norm.
        not_onbounds = (optret.x!=0) * (optret.x!=1)
        gnorm = max(np.abs(optret.jac)*not_onbounds)

        print("fdiff: %f"%fdiff)
        print("gnorm: %f"%gnorm)

        it += 1

    ides = init_design

    if optret.success:
        msg = "Successful Search Termination"
    else:
        msg = "Abnormal Search Termination"
    print(''.join([msg, ". Final Entropy = %f"%(-optret.fun)]))
    X_sol = optret.x.reshape([N,P])

    Dmat = distance_matrix(X_sol, X_sol)
    np.min(Dmat[np.triu_indices(N, k = 1)])

    return {'design' : X_sol, 'entropy' : -float(optret.fun),  'optret' : optret,
            'init_design' : init_design, 'init_entropy' : -spy_maxent_cost(init_design), 'nnet' : model}
