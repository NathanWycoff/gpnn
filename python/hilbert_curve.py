#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/hilbert_curve.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 03.27.2019

## Initialize a design along a hilbert curve. 

import matplotlib.pyplot as plt
plt.ion()
from hilbertcurve.hilbertcurve import HilbertCurve
import numpy as np

def hc_design(N, P):
    """
    Design along a Hilbert Curve in the P dimensional unit box with N many points.
    """
    gamma = int(np.ceil(np.log2(N+1) / P))
    hilbert_curve = HilbertCurve(gamma, P)

    n_verts = 2**(gamma*P) - 1
    cube_slen = float(np.power(2, gamma))

    hc = hilbert_curve.coordinates_from_distance

    design = np.empty([N,P])
    for n in range(N):
        ind = float(n * n_verts) / float(N)
        xb = np.array(hc(int(np.floor(ind)))) / cube_slen 
        xa = np.array(hc(int(np.ceil(ind)))) / cube_slen
        conv_coef = ind - np.floor(ind)
        design[n,:] = (1.0 - conv_coef) * xb + conv_coef * xa

    return(design)
