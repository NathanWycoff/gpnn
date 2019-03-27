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

tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels
kernel = psd_kernels.ExponentiatedQuadratic(length_scale = np.array([0.1]).astype(np.float64))

def bump(x, lb, ub, scale = 10):
    """
    A smooth function which will be large between lb and ub, and zero elsewhere.

    :param x: The function argument.
    :param lb, ub: Function will be nonzero only within lb, ub.
    :param scale: Max value of function.
    """
    xn11 = 2 * (x-lb) / (ub - lb) - 1
    return scale * np.e * tf.exp(-1/(1-tf.square(tf.clip_by_value(xn11,-1,1))))


#TODO: Make this a class to be more pythonic?
def neural_maxent(N, P, L, H, R, minalldist = 1e-5, scalealldist = 1500):
    """
    Create a neural maxentropy design with N points in P dimensions.
    """

    def loss(X):
        #### Compute low D subspace.
        Z = model(X)

        #### Small Distance penalty.
        r = tf.reduce_sum(X*X, 1)

        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        D = r - 2*tf.matmul(X, tf.transpose(X)) + tf.transpose(r)

        Da = D + tf.cast(np.power(P, 2) * tf.eye(N), np.float64)
        mindist = tf.reduce_min(Da)
        distpen = bump(mindist, -minalldist, minalldist, scalealldist)

        # Detect if near boundary
        #bump(X, )
        
        # Get the entropy of the design
        gp = tfd.GaussianProcess(kernel, Z, jitter = 1E-10)
        nldetK = -tf.linalg.logdet(gp.covariance())

        return nldetK + distpen


    init_design = hc_design(N,P)
    X = tf.Variable(init_design)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten()] + 
        [tf.keras.layers.Dense(H, activation=tf.nn.tanh, input_shape=(P,)) for _ in range(L)] + 
        [tf.keras.layers.Dense(R, input_shape = (P,))
        ])

    ### With SCIPY BFGS
    def scipy_cost(x):
        """
        x is the vector of inputs, while X is a tensorflow matrix of appropriate size.
        """
        X.assign(np.array(x).reshape([N,P]))
        return loss(X).numpy()
    def scipy_grad(x):
        X.assign(np.array(x).reshape([N,P]))
        with tf.GradientTape() as t:
            l = loss(X)
        return (t.gradient(l, X).numpy()).flatten()

    optret = minimize(scipy_cost, init_design, bounds = [(0,1) for _ in range(N*P)], method = 'L-BFGS-B',\
            jac = scipy_grad, options = {'ftol' : 0})
    ides = init_design

    print(optret.success)
    print(optret.fun)
    X_sol = optret.x.reshape([N,P])

    Dmat = distance_matrix(X_sol, X_sol)
    np.min(Dmat[np.triu_indices(N, k = 1)])

    return {'design' : X_sol, 'entropy' : -float(optret.fun),  'optret' : optret,
            'init_design' : init_design, 'init_entropy' : -scipy_cost(init_design), 'nnet' : model}

