#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/preimage.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.03.2019
# Can we find the preimage of the level set?


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
tf.enable_eager_execution()
from scipy.optimize import linprog

P = 5
L = 1
H = 10
R = 2

tf.random.set_random_seed(123)
np.random.seed(123)
act = tf.nn.relu
model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(H, activation=act, input_shape=(P,)) if i == 0 else tf.keras.layers.Dense(H, activation=act) for i in range(L)] + 
    [tf.keras.layers.Dense(R)
    ])
model.build(input_shape=[P])
model.summary()

x = np.random.normal(size=[1,P])
z = model(tf.cast(x, tf.float32)).numpy()

W1 = model.get_weights()[0].T
W2 = model.get_weights()[2].T

W1d = np.linalg.pinv(W1)
W2d = np.linalg.pinv(W2)

W2d.dot(z.T)

c = np.zeros([W2.shape[1]])
hstar = linprog(c, A_eq = W2, b_eq = z).x
xstar = W1d.dot(hstar)

model(tf.cast(x, tf.float32))
model(tf.cast(xstar.reshape([1,P]), tf.float32))

h1 = W1.dot(x.T)
h2 = tf.nn.relu(h1).numpy()
h2 = (h1 > 0) * h1
W2.dot(h2)
