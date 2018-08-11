# -*- coding: utf-8 -*-
from __future__ import division

import sys
import numpy as np

# python 3.6 comparability
if sys.version_info < (3, 0):
    srange = xrange
else:
    srange = range


# ---- Correlation Function ----
def exponential_ems(n_spec, nr, wvnb, kl, corrlength):
    """
    Correlation Functions for I2EM Emissivity Model.

    Parameters
    ----------
    n_spec : array_like
        Spectrum range.
    nr : float
        Length of n_spec.
    wvnb : float
        Calculated by SurfScat Module.
    kl : float or array_like
        Multiplication of correlation length with wave number
    corrlength : float
        Correlation length in cm.
    """

    wn = np.zeros([n_spec, nr])

    Wn = []
    for n in srange(n_spec):
        wn[n, :] = (n + 1) * kl ** 2 / ((n + 1) ** 2 + (wvnb * corrlength) ** 2) ** 1.5

    return wn


def gaussian_ems(n_spec, nr, wvnb, kl, corrlength):
    """
    Correlation Functions for I2EM Emissivity Model.

    Parameters
    ----------
    n_spec : array_like
        Spectrum range.
    nr : float
        Length of n_spec.
    wvnb : float
        Calculated by SurfScat Module.
    kl : float or array_like
        Multiplication of correlation length with wave number
    corrlength : float
        Correlation length in cm.
    """
    wn = np.zeros([n_spec, nr])

    for n in srange(n_spec):
        wn[n, :] = 0.5 * kl ** 2 / (n + 1) * np.exp(-(wvnb * corrlength) ** 2 / (4 * (n + 1)))

    return wn


def mixed_ems(n_spec, nr, wvnb, kl, corrlength):
    """
    Correlation Functions for I2EM Emissivity Model.

    Parameters
    ----------
    n_spec : array_like
        Spectrum range.
    nr : float
        Length of n_spec.
    wvnb : float
        Calculated by SurfScat Module.
    kl : float or array_like
        Multiplication of correlation length with wave number
    corrlength : float
        Correlation length in cm.
    """
    gauss = gaussian_ems(n_spec, nr, wvnb, kl, corrlength)
    exp = exponential_ems(n_spec, nr, wvnb, kl, corrlength)

    wn = gauss / exp

    return wn
