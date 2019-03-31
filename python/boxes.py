#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/boxes.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 03.28.2019

## Make bobby's boxes.

from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.ion()

N = 10
P = 2
X = np.random.uniform(size=[N,P])

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
distances /= 2


fig = plt.figure()

plt.scatter(X[:,0], X[:,1])

#for n in range(N):
#    d = distances[n][1]
#    matplotlib.patches.Rectangle((X[n,0] - d/2.0, X[n,1] - d/2.0), d, d)
plt.scatter(X[:,0] + [d[1] for d in distances], X[:,1])
plt.scatter(X[:,0] - [d[1] for d in distances], X[:,1])
plt.scatter(X[:,0], X[:,1] + [d[1] for d in distances])
plt.scatter(X[:,0], X[:,1] - [d[1] for d in distances])

plt.savefig('temp.pdf')
