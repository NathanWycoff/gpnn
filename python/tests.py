#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/tests.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 03.27.2019

exec(open("python/hilbert_curve.py").read())
exec(open("python/neural_maxent.py").read())

#######################################################
#######################################################
# Visual Testing ######################################
#######################################################
#######################################################

#############################################
## Visualize a Hilbert Curve Design in 2D.
N = 100#number of desired design points
P = 2#Dimensionality of design space.

init_design = hc_design(N, P)

fig = plt.figure(figsize = [8,8])
plt.scatter(init_design[:,0], init_design[:,1])
plt.show()

plt.savefig('temp.pdf')

#############################################
## Test that neural entropy matches standard entropy on a standard problem.
N = 100
P = 100
L = 2
H = 10
R = 2
# TODO: These next two dials are pretty fragile.
minalldist = 1e-5
scalealldist = 1500

ne = neural_maxent(N ,P, L, H, R)

fig = plt.figure(figsize = [8,8])
ides = ne['init_design']
X_sol = ne['design']
model = ne['nnet']
optret = ne['optret']
icost = ne['init_entropy']

plt.subplot(2,2,1)
plt.scatter(ides[:,0], ides[:,1])
plt.title("Initial Design (cost %s)"%np.around(icost, decimals = 2))

plt.subplot(2,2,2)
plt.scatter(X_sol[:,0], X_sol[:,1])
plt.title("Final Design (cost %s)"%np.around(optret.fun))

Z_sol = model(ides)
plt.subplot(2,2,3)
if R == 1:
    plt.scatter([0 for _ in range(N)], Z_sol[:])
elif R == 2:
    plt.scatter(Z_sol[:,0], Z_sol[:,1])
plt.title("Initial Latent Representation")

Z_sol = model(X_sol)
plt.subplot(2,2,4)
if R == 1:
    plt.scatter([0 for _ in range(N)], Z_sol[:])
elif R == 2:
    plt.scatter(Z_sol[:,0], Z_sol[:,1])
plt.title("Final Latent Representation")

plt.show()

plt.savefig('temp.pdf')
