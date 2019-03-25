#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/playground.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 03.21.2019

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels

P = 2 # Size of design space
R = 1 # Dim of Latent space.
H = 1 # Number of hidden units
N = 10 # Sample size

# Set up a model with random init
model = Sequential()
model.add(Dense(units=H, activation='relu', input_dim=P))
model.add(Dense(units=R, activation='linear'))

model.compile(loss='mse',
              optimizer='sgd')

X = np.random.normal(size=[N,P]).astype(np.float32)
Z = model.predict(X)

kernel = psd_kernels.ExponentiatedQuadratic()

gp = tfd.GaussianProcess(kernel, Z)

tf.linalg.logdet(gp.covariance())

def obj(X):
    #Z = model(tf.cast(X, np.float32))
    print('a')
    Z = model(tf.math.sigmoid(X))

    kernel = psd_kernels.ExponentiatedQuadratic()

    gp = tfd.GaussianProcess(kernel, Z)

    nldetK = -tf.linalg.logdet(gp.covariance())
    return nldetK, tf.gradients(nldetK, X)[0]

optim_results = tfp.optimizer.bfgs_minimize(
  obj, initial_position=tf.constant(X), tolerance=1e-8)

init_op = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init_op)
    results = session.run(optim_results)
    print ("Objective Value %d" % results.objective_value)
    print ("Optimizing X:")
    print (results.position)
    print ("Function evaluations: %d" % results.num_objective_evaluations)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/playground.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 03.21.2019

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow_probability as tfp
import tensorflow as tf
tf.enable_eager_execution()

tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels

P = 2 # Size of design space
R = 1 # Dim of Latent space.
H = 64 # Number of hidden units
N = 10 # Sample size


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(H, activation=tf.nn.relu, input_shape=(P,)),
    tf.keras.layers.Dense(R)
    ])

with tf.GradientTape() as t:
    X = np.random.normal(size=[N,P])
    X_tf = tf.math.sigmoid(tf.contrib.eager.Variable(tf.to_float(X)))
    t.watch(X_tf)

    Z = model(X_tf)

    kernel = psd_kernels.ExponentiatedQuadratic()

    gp = tfd.GaussianProcess(kernel, Z)

    nldetK = -tf.linalg.logdet(gp.covariance())

# Derivative of z with respect to the original input tensor x
dnldetK_dX = t.gradient(nldetK, X_tf)

def obj(X):
    Z = model(X_tf)

    kernel = psd_kernels.ExponentiatedQuadratic()

    gp = tfd.GaussianProcess(kernel, Z)

    nldetK = -tf.linalg.logdet(gp.covariance())
    return nldetK, tf.gradients(nldetK, X)[0]


optim_results = tfp.optimizer.bfgs_minimize(
  obj, initial_position=X, tolerance=1e-8)
