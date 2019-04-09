#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/test_funcs.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.09.2019

# Some functions to test my shit on

#########
def myackley(z, extent):
    assert R == 2
    z[0] = (z[0] - extent[0][0]) / (extent[0][1] - extent[0][0])
    z[1] = (z[1] - extent[1][0]) / (extent[1][1] - extent[1][0])
    return(ackley(z))

## Get a design
def neural_ackley(x, model, extent):
    """
    An ackley defined on a low D space.
    """
    xs = x.reshape([1,x.shape[0]])
    z = model(tf.cast(xs, tf.float32)).numpy().reshape(R)
    # Reshape according to the extent of the low D points.
    return(myackley(z, extent))
