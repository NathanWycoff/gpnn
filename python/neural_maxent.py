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
exec(open("python/hilbert_curve.py").read())

tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels
kernel = psd_kernels.ExponentiatedQuadratic(amplitude = np.array([1]).astype(np.float32), length_scale = np.array([0.1]).astype(np.float32))

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
def neural_maxent(N, P, L, H, R, minalldist = 1e-5, scalealldist = 1500, net_weights = None):
    """
    Create a neural maxentropy design with N points in P dimensions.

    :param N: The number of initial design points
    :param P: The dimension of the input space.
    :param L: The number of layers in the neural net.
    :param H: Width of hidden layers (scalar).
    :param R: The dimension of the output space.
    :param net_weights: The weights to initialize the neural net, to be passed to the set_weights method of the neural net object.

    """

    def maxent_loss(X):
        #### Compute low D subspace.
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
        
        # Get the entropy of the design
        gp = tfd.GaussianProcess(kernel, Z, jitter = 1E-4)
        nldetK = -tf.linalg.logdet(gp.covariance())

        return nldetK + distpen


    init_design = hc_design(N,P)
    X = tf.Variable(init_design)

    act = tf.nn.tanh
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(H, activation=act, input_shape=(P,)) if i == 0 else tf.keras.layers.Dense(H, activation=act) for i in range(L)] + 
        [tf.keras.layers.Dense(R)
        ])
    model.build(input_shape=[P])
    if net_weights is not None:
        model.set_weights(net_weights)

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

    optret = minimize(spy_maxent_cost, init_design, bounds = [(0,1) for _ in range(N*P)], method = 'L-BFGS-B',\
            jac = spy_maxent_grad)
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
        gp = tfd.GaussianProcess(kernel, Z, jitter = 1E-10)
        nldetK = -tf.linalg.logdet(gp.covariance())

        return nldetK + distpen


    init_design = hc_design(N,P)
    X = tf.Variable(init_design)

    act = tf.nn.tanh
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(H, activation=act, input_shape=(P,)) if i == 0 else tf.keras.layers.Dense(H, activation=act) for i in range(L)] + 
        [tf.keras.layers.Dense(R)
        ])
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
