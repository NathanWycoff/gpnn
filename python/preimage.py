#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/preimage.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.03.2019
# Can we find the preimage of the level set?


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

tf.random.set_random_seed(123)
np.random.seed(123)
act = tf.nn.tanh
model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(H, activation=act, input_shape=(P,)) if i == 0 else tf.keras.layers.Dense(H, activation=act) for i in range(L)] + 
    [tf.keras.layers.Dense(R)
    ])
model.build(input_shape=[P])
model.summary()

