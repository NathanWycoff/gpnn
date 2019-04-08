#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/test_functions.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.08.2019

#########
#TODO: WARNING -- uses global variable extent (terrible practice).
def myackley(z):
    assert R == 2
    z[0] = (z[0] - extent[0]) / (extent[1] - extent[0])
    z[1] = (z[1] - extent[2]) / (extent[3] - extent[2])
    return(ackley(z))

## Get a design
def ackley_obj(x):
    """
    An ackley defined on a low D space.
    """
    xs = x.reshape([1,x.shape[0]])
    z = model(tf.cast(xs, tf.float32)).numpy().reshape(R)
    # Reshape according to the extent of the low D points.
    return(myackley(z))

