#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/opt_lib.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.01.2019

################ Contains acquisition functions for BO
def get_var(xx_tf, yn_tf, gp):
    """
    :param xx_tf: A tensor giving the new point to evaluate at.
    :param yn_tf: A tensor giving all previously observed responses.
    :param gp: A gp used to predict. GP should be trained on the locations yn_tf was observed.
    """

    N, P = gp.index_points.numpy().shape
    k = gp.kernel
    kxx = tf.reshape(k.apply(xx_tf, gp.index_points), [N,1])
    K = tf.squeeze(gp.covariance())
    Kl = tf.squeeze(tf.linalg.cholesky(gp.covariance()))
    alpha = tf.cast(tf.linalg.solve(tf.cast(tf.transpose(Kl), tf.float64), tf.linalg.solve(tf.cast(Kl, tf.float64), yn_tf)), tf.float32)
    v = tf.linalg.solve(Kl, kxx)

    zpred_mean = tf.squeeze(tf.matmul(tf.transpose(kxx), alpha))
    kkxx = kernel.apply(xx_tf, xx_tf)
    zpred_vars = tf.squeeze(kkxx - tf.matmul(tf.transpose(v),v))

    return(zpred_vars)

def spy_nvar(x, model, gp, response_tf):
    P = len(x)
    x_tf = tf.cast(tf.Variable(x.reshape([1,P])), tf.float32)
    z_tf = model(x_tf)
    #z_tf = tf.cast(tf.Variable(z.reshape([1,R])), tf.float32)
    ei = get_var(z_tf, response_tf, gp)
    return(-float(ei.numpy().flatten()))

def spy_nvar_grad(x, model, gp, response_tf):
    P = len(x)
    with tf.GradientTape() as t:
        x_tf = tf.cast(tf.Variable(x.reshape([1,P])), tf.float32)
        z_tf = model(x_tf)
        #z_tf = tf.cast(tf.Variable(z.reshape([1,R])), tf.float32)
        ei = get_var(z_tf, response_tf, gp)
    grad = t.gradient(ei, x_tf)
    return(-grad.numpy().flatten().astype(np.float64))

def get_ei(xx_tf, yn_tf, gp):
    """
    :param xx_tf: A tensor giving the new point to evaluate at.
    :param yn_tf: A tensor giving all previously observed responses.
    :param gp: A gp used to predict. GP should be trained on the locations yn_tf was observed.
    """

    N, P = gp.index_points.numpy().shape
    k = gp.kernel
    kxx = tf.reshape(k.apply(xx_tf, gp.index_points), [N,1])
    K = tf.squeeze(gp.covariance())
    Kl = tf.squeeze(tf.linalg.cholesky(gp.covariance()))
    alpha = tf.cast(tf.linalg.solve(tf.cast(tf.transpose(Kl), tf.float64), tf.linalg.solve(tf.cast(Kl, tf.float64), yn_tf)), tf.float32)
    v = tf.linalg.solve(Kl, kxx)

    zpred_mean = tf.squeeze(tf.matmul(tf.transpose(kxx), alpha))
    kkxx = kernel.apply(xx_tf, xx_tf)
    zpred_vars = tf.squeeze(kkxx - tf.matmul(tf.transpose(v),v))

    miny = tf.cast(tf.reduce_min(yn_tf), tf.float32)

    pdist = tfp.distributions.Normal(tf.squeeze(zpred_mean), tf.squeeze(tf.sqrt(zpred_vars)))
    #pdist = tfp.distributions.Normal(tf.squeeze(zpred_mean), tf.squeeze((zpred_vars)))
    ei = (miny - zpred_mean) * pdist.cdf(miny) + \
            zpred_vars * pdist.prob(miny)
    return(ei)

def spy_neur_nei(x, model, gp, response_tf):
    P = len(x)
    x_tf = tf.cast(tf.Variable(x.reshape([1,P])), tf.float32)
    z_tf = model(x_tf)
    #z_tf = tf.cast(tf.Variable(z.reshape([1,R])), tf.float32)
    ei = get_ei(z_tf, response_tf, gp)
    return(-float(ei.numpy().flatten()))

def spy_neur_nei_grad(x, model, gp, response_tf):
    P = len(x)
    with tf.GradientTape() as t:
        x_tf = tf.cast(tf.Variable(x.reshape([1,P])), tf.float32)
        z_tf = model(x_tf)
        #z_tf = tf.cast(tf.Variable(z.reshape([1,R])), tf.float32)
        ei = get_ei(z_tf, response_tf, gp)
    grad = t.gradient(ei, x_tf)
    return(-grad.numpy().flatten().astype(np.float64))

#def get_ei(z_tf):
#    kxx = tf.reshape(k.apply(z_tf, Z), [N,1])
#    K = tf.squeeze(gp.covariance())
#    Kl = tf.squeeze(tf.linalg.cholesky(gp.covariance()))
#    alpha = tf.cast(tf.linalg.solve(tf.cast(tf.transpose(Kl), tf.float64), tf.linalg.solve(tf.cast(Kl, tf.float64), response_tf)), tf.float32)
#    v = tf.linalg.solve(Kl, kxx)
#
#    zpred_mean = tf.matmul(tf.transpose(kxx), alpha)
#    zpred_vars = k.amplitude - tf.matmul(tf.transpose(v),v)
#
#    miny = min(response)
#
#    pdist = tfp.distributions.Normal(tf.squeeze(zpred_mean), tf.squeeze(tf.sqrt(zpred_vars)))
#    #pdist = tfp.distributions.Normal(tf.squeeze(zpred_mean), tf.squeeze((zpred_vars)))
#    ei = (miny - zpred_mean) * pdist.cdf(miny) + \
#            zpred_vars * pdist.prob(miny)
#    return(ei)

######### Weight estimation
def weights_2_vec(weights):
    return(np.concatenate([wx.flatten() for wx in weights]))

def vec_2_weights(vec, model):
    nlayers = L+1
    weights = []
    used_neurons = 0
    prev_shape = P
    for l in range(nlayers):
        # Add connection weights:
        curr_size = model.layers[l].output_shape[1] * prev_shape
        weights.append(vec[used_neurons:(used_neurons+curr_size)].reshape([ prev_shape, model.layers[l].output_shape[1]]))
        used_neurons += curr_size

        # Add biases:
        curr_size = model.layers[l].output_shape[1] 
        vec[used_neurons:(used_neurons+curr_size)]
        weights.append(vec[used_neurons:(used_neurons+curr_size)])
        used_neurons += curr_size

        prev_shape = model.layers[l].output_shape[1]

    return(weights)

# Define likelihood wrt weights
def weights_loss(weights, model, design, response):
    """
    Returns the negative log likelihood (divided by sample size).
    """
    model.set_weights(weights)
    Z = model(tf.cast(design, np.float32))

    # Get the entropy of the design
    gp = tfd.GaussianProcess(kernel, Z, jitter = 1E-4)
    nll = -gp.log_prob(response) / float(design.shape[0])

    return nll

def spy_weights_cost(w, model, design, response, l2_coef):
    weights = vec_2_weights(w, model)
    nll = weights_loss(weights, model, design, response)
    l2_reg = tf.add_n([ tf.nn.l2_loss(v) for v in model.trainable_weights
                    if 'bias' not in v.name ]) * l2_coef
    ret = nll + l2_reg
    return float(ret.numpy())

def spy_weights_grad(w, model, design, response, l2_coef):
    weights = vec_2_weights(w, model)
    with tf.GradientTape() as t:
        nll = weights_loss(weights, model, design, response)
        l2_reg = tf.add_n([ tf.nn.l2_loss(v) for v in model.trainable_weights
                        if 'bias' not in v.name ]) * l2_coef
        ret = nll + l2_reg
    dweights = t.gradient(ret, model.trainable_weights)
    dweightsnp = [wi.numpy().astype(np.float64) for wi in dweights]

    return weights_2_vec(dweightsnp)

def update_weights(design, response, model, l2_coef = 1):
    init_w = weights_2_vec(model.get_weights()).astype(np.float64)
    optret = minimize(spy_weights_cost, init_w, method = 'L-BFGS-B',\
            jac = spy_weights_grad, args = (model, design, response, l2_coef))
    model.set_weights(vec_2_weights(optret.x, model))
    if optret.success:
        msg = "Successful Weight Opt Termination"
    else:
        msg = "Abnormal Weight Opt Termination"
    return(model)
