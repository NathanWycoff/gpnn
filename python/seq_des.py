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
exec(open("python/hilbert_curve.py").read())
exec(open("python/ackley.py").read())

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

f = ackley

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten()] + 
    [tf.keras.layers.Dense(H, activation=tf.nn.tanh, input_shape=(P,)) for _ in range(L)] + 
    [tf.keras.layers.Dense(R, input_shape = (P,))
    ])

