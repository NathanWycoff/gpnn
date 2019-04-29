#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/tests.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 03.27.2019

from sklearn.manifold import MDS
import matplotlib.pyplot as plt
plt.ion()
exec(open("python/hilbert_curve.py").read())
exec(open("python/opt_lib.py").read())
exec(open("python/neural_maxent.py").read())
exec(open("python/misc.py").read())

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
#TODO: ## Test that neural entropy matches standard entropy on a standard problem.
#### Visual check
N = 100
P = 10
L = 10
H = 30
R = 2
# TODO: These next two dials are pretty fragile.
minalldist = 1e-5
scalealldist = 1500

ne = neural_maxent(N ,P, L, H, R)

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

Z_sol = model(tf.cast(ides, tf.float32))
plt.subplot(2,2,3)
if R == 1:
    plt.scatter([0 for _ in range(N)], Z_sol[:])
elif R == 2:
    plt.scatter(Z_sol[:,0], Z_sol[:,1])
plt.title("Initial Latent Representation")

Z_sol = model(tf.cast(X_sol, tf.float32))
plt.subplot(2,2,4)
if R == 1:
    plt.scatter([0 for _ in range(N)], Z_sol[:])
elif R == 2:
    plt.scatter(Z_sol[:,0], Z_sol[:,1])
plt.title("Final Latent Representation")

plt.show()

plt.savefig('temp.pdf')

#############################################
## Test that the sequential max-entropy seems to be doing something reasonable.
tf.random.set_random_seed(1234)
np.random.seed(1234)

N_init = 4
P = 2
L = 0
H = None
R = 2
seq_steps = 30

# Two similarly shaped random nets, one is the one we init on, one the one we use.
model = random_nn(P,L,H,R, act = tf.nn.tanh)
model.set_weights([0.1*np.eye(2), np.zeros(2)])

# Make the initial design
init_w = model.get_weights()
design = neural_maxent(N_init, P, L, H, R, net_weights = model.get_weights())['design']

# Add a point
new_point = neural_maxent(10, P, L, H, R, X_prev = design, net_weights = model.get_weights())['design']

fig = plt.figure()
plt.scatter(design[:,0], design[:,1])
plt.scatter(new_point[:,0], new_point[:,1])
plt.savefig('temp.pdf')


#############################################
## Test that EI is working by reproducing an images from Bobby's notes.
N = 5
x = np.array([1,2,3,4,12])
y = np.array([0,-1.75,-2,-0.5,5])
x_tf = tf.cast(tf.Variable(x.reshape([N,1])), tf.float32)
y_tf = tf.Variable(y.reshape([N,1]))

# Initial fit to get amplitude (this is horribly inefficient but easiest for now)
kernel = psd_kernels.ExponentiatedQuadratic(amplitude = np.array([1]).astype(np.float32), length_scale = np.array([2.23]).astype(np.float32))
gp = tfd.GaussianProcess(kernel, x_tf, jitter = tf.cast(tf.Variable(1E-6), tf.float32))

# Plug in estimate
K = tf.squeeze(gp.covariance())
tau_hat = tf.squeeze(tf.sqrt(tf.matmul(tf.transpose(y_tf), tf.linalg.solve(tf.cast(K, tf.float64), y_tf)) / float(N)))
kernel = psd_kernels.ExponentiatedQuadratic(amplitude = np.array([tau_hat]).astype(np.float32), length_scale = np.array([2.23]).astype(np.float32))
gp = tfd.GaussianProcess(kernel, x_tf, jitter = tf.cast(tf.Variable(1E-6), tf.float32))

# Determine kernel
K = tf.squeeze(gp.covariance())
Kl = tf.squeeze(tf.linalg.cholesky(gp.covariance()))
alpha = tf.cast(tf.linalg.solve(tf.cast(tf.transpose(Kl), tf.float64), tf.linalg.solve(tf.cast(Kl, tf.float64), y_tf)), tf.float32)

def get_ei(xx_tf, yn_tf, gp):
    """
    :param xx_tf: A tensor giving the new point to evaluate at.
    :param yn_tf: A tensor giving all previously observed responses.
    :param gp: A gp used to predict. GP should be trained on the locations yn_tf was observed.
    """

    k = gp.kernel
    kxx = tf.reshape(k.apply(xx_tf, gp.index_points), [N,1])
    K = tf.squeeze(gp.covariance())
    Kl = tf.squeeze(tf.linalg.cholesky(gp.covariance()))
    alpha = tf.cast(tf.linalg.solve(tf.cast(tf.transpose(Kl), tf.float64), tf.linalg.solve(tf.cast(Kl, tf.float64), yn_tf)), tf.float32)
    v = tf.linalg.solve(Kl, kxx)

    zpred_mean = tf.squeeze(tf.matmul(tf.transpose(kxx), alpha))
    kkxx = kernel.apply(xxn_tf, xxn_tf)
    zpred_vars = tf.squeeze(kkxx - tf.matmul(tf.transpose(v),v))

    miny = tf.cast(tf.reduce_min(yn_tf), tf.float32)

    pdist = tfp.distributions.Normal(tf.squeeze(zpred_mean), tf.squeeze(tf.sqrt(zpred_vars)))
    #pdist = tfp.distributions.Normal(tf.squeeze(zpred_mean), tf.squeeze((zpred_vars)))
    ei = (miny - zpred_mean) * pdist.cdf(miny) + \
            zpred_vars * pdist.prob(miny)
    return(ei)

NN = 100
xx = np.linspace(0, 13, NN)
zpred_means = np.empty([NN])
zpred_vars = np.empty([NN])
eis = np.empty([NN])
for nn in range(NN):
    xxn = xx[nn].reshape([1,1])
    xxn_tf = tf.cast(tf.Variable(xxn), tf.float32)

    kxx = tf.reshape(kernel.apply(xxn_tf, x_tf), [N,1])
    v = tf.linalg.solve(Kl, kxx)

    zpred_means[nn] = tf.matmul(tf.transpose(kxx), alpha).numpy()
    kkxx = kernel.apply(xxn_tf, xxn_tf)
    zpred_vars[nn] = (kkxx - tf.matmul(tf.transpose(v),v)).numpy()
    eis[nn] = get_ei(xxn_tf, y_tf, gp)

fig = plt.figure(figsize = [8,8])

plt.subplot(1,2,1)
plt.scatter(x, y)
plt.plot(xx, zpred_means)
plt.plot(xx, zpred_means + 2 * np.sqrt(zpred_vars))
plt.plot(xx, zpred_means - 2 * np.sqrt(zpred_vars))

plt.subplot(1,2,2)
plt.plot(xx, eis)

plt.show()
plt.savefig('temp.pdf')

### test that the vanilla GP seems to work.
N = 1000
P = 6
X = np.random.uniform(size=[N,P]).astype(np.float64)

kernel = psd_kernels.ExponentiatedQuadratic(amplitude = np.array([4.20]).astype(np.float64), length_scale = np.array([0.1]).astype(np.float64))
gp = tfd.GaussianProcess(kernel, X, jitter = 1E-4)
covmat = gp.covariance().numpy().squeeze()

L = np.linalg.cholesky(covmat)
z = np.random.normal(size=[N])
y = L.dot(z) + 10

#try to recover those params.
van_model = random_nn(P,0,None,P)
van_model.set_weights([0.1*np.eye(P), np.repeat(np.mean(y), P)])

van_model = update_weights(X, y, van_model, l2_coef = 0, diag = True)[0]
van_model.get_weights()
## Formula to go from diagonal of weight element to length-scale is 1/w * 1/sqrt(2)
## Looks pretty close

#TODO: Kernel handling is sloppy; make it less so. (i.e. we define a global one, we tend to use arbitrary amplitude).
#TODO: We have different nuggets in different places; that should not be so.
