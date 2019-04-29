#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/ackley_bakeoff_anlz.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.21.2019

## Analyze output from ackley_bakeoff.py
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

nps = np.load('./data/emb_quad_bakeoff_2_100_2.npz')

seq_steps = len(nps['nei_min_obj'])

fig = plt.figure()
plt.plot(np.apply_along_axis(np.median, 1, nps['nei_min_obj']))
plt.plot(np.apply_along_axis(np.median, 1, nps['nme_min_obj']))
plt.plot(np.apply_along_axis(np.median, 1, nps['van_min_obj']))
plt.legend(['NEI', 'NME', 'VAN'])
plt.savefig("./images/ack_min_obj.pdf")

fig = plt.figure()
plt.plot(np.apply_along_axis(np.median, 1, nps['nei_weight_diff']))
plt.plot(np.apply_along_axis(np.median, 1, nps['nme_weight_diff']))
plt.savefig("./images/ack_weight_diff.pdf")

fig = plt.figure()
plt.plot(np.apply_along_axis(np.median, 1, nps['nei_wup_iters']))
plt.plot(np.apply_along_axis(np.median, 1, nps['nme_wup_iters']))
plt.savefig("./images/ack_wup_iters.pdf")
