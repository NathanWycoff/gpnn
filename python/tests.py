#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/tests.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 03.27.2019

exec(open("python/hilbert_curve.py").read())

#############################################
## Visualize a Hilbert Curve Design in 2D.
N = 100#number of desired design points
P = 2#Dimensionality of design space.

init_design = hc_design(N, P)

fig = plt.figure(figsize = [8,8])
plt.scatter(init_design[:,0], init_design[:,1])
plt.show()

plt.savefig('temp.pdf')
