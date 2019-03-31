#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/box_vs_bump.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 03.28.2019

## The maxent problem is prone to get caught in what I believe may be saddle points of the Lagrangian. 
## Bobby and I have different methods to resolve this. Here, we will compare them.

from sklearn.manifold import MDS
exec(open("python/hilbert_curve.py").read())
exec(open("python/neural_maxent.py").read())

#############################################
#TODO: ## Test that neural entropy matches standard entropy on a standard problem.
#### Visual check
N = 100
P = 2
L = 10
H = 10
R = 2
# TODO: These next two dials are pretty fragile.
minalldist = 1e-5
scalealldist = 1500

#tf.random.set_random_seed(123)

ne = neural_maxent(N ,P, L, H, R)
#ne = neural_maxent_box(N ,P, L, H, R, outer_bfgs_iters = 1001)

ides = ne['init_design']
X_sol = ne['design']
model = ne['nnet']
optret = ne['optret']
icost = ne['init_entropy']
cost = ne['entropy']

embedding = MDS()

fig = plt.figure(figsize = [8,8])
plt.subplot(2,2,1)
if P > 2:
    toplot = embedding.fit_transform(ides)
    mdsmessage = "(MDS)"
else:
    toplot = ides
    mdsmessage = ""
plt.scatter(toplot[:,0], toplot[:,1])
plt.title(''.join(["Initial Design ", mdsmessage, " (cost %s)"%np.around(icost, decimals = 2)]))

plt.subplot(2,2,2)
if P > 2:
    toplot = embedding.fit_transform(X_sol)
    mdsmessage = "(MDS)"
else:
    toplot = X_sol
    mdsmessage = ""
plt.scatter(toplot[:,0], toplot[:,1])
plt.title(''.join(["Final Design ", mdsmessage, " (cost %s)"%np.around(cost, decimals = 2)]))

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
