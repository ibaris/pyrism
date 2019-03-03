# -*- coding: utf-8 -*-
"""
Created on 03.03.2019 by Ismail Baris
"""
from __future__ import division

import sys

import numpy as np
from scipy.special import kv, gamma

# python 3.6 comparability
if sys.version_info < (3, 0):
    srange = xrange
else:
    srange = range


# ---- Correlation Function ----
def exponential(sigma, corrlength, wvnb, Ts, n=None):
    """
    Correlation Functions for I2EM Model.

    Parameters
    ----------
    n : int (>1)
        Coefficient needed for x-power and x-exponential
        correlation function
    wvnb : float
        Calculated by SurfScat Module.
    corrlength : int or float
        Correlation length (cm)
    sigma : int or float
        RMS Height (cm)
    Ts : float
        Calculated by SurfScat Module.
    """

    Wn = []
    for i in srange(1, Ts + 1):
        # i += 1
        wn = (corrlength ** 2 / i ** 2) * (1 + (wvnb * corrlength / i) ** 2) ** (-1.5)
        Wn.append(wn)

    Wn = np.asarray(Wn, dtype=np.double)
    rss = sigma / corrlength

    return Wn, rss


def gaussian(sigma, corrlength, wvnb, Ts, n=None):
    """
    Correlation Functions for I2EM Model.

    Parameters
    ----------
    n : int (>1)
        Coefficient needed for x-power and x-exponential
        correlation function
    wvnb : float
        Calculated by SurfScat Module.
    corrlength : int or float
        Correlation length (cm)
    sigma : int or float
        RMS Height (cm)
    Ts : float
        Calculated by SurfScat Module.
    """
    Wn = []
    for i in srange(Ts):
        i += 1
        wn = corrlength ** 2 / (2 * i) * np.exp(-(wvnb * corrlength) ** 2 / (4 * i))
        Wn.append(wn)

    Wn = np.asarray(Wn, dtype=np.float)
    rss = np.sqrt(2) * sigma / corrlength

    return Wn, rss


def xpower(sigma, corrlength, wvnb, Ts, n):
    """
    Correlation Functions for I2EM Model.

    Parameters
    ----------
    n : int (>1)
        Coefficient needed for x-power and x-exponential
        correlation function
    wvnb : float
        Calculated by SurfScat Module.
    corrlength : int or float
        Correlation length (cm)
    sigma : int or float
        RMS Height (cm)
    Ts : float
        Calculated by SurfScat Module.
    """
    Wn = []
    for i in srange(Ts):
        i += 1
        wn = corrlength ** 2 * (wvnb * corrlength) ** (-1 + n * i) * kv(
            1 - n * i, wvnb * corrlength) / (2 ** (n * i - 1) * gamma(n * i))
        Wn.append(wn)

    Wn = np.asarray(Wn, dtype=np.float)
    if n == 1.5:
        rss = np.sqrt(n * 2) * sigma / corrlength
    else:
        rss = 0

    return Wn, rss


def mixed(sigma, corrlength, wvnb, Ts, n):
    """
    Correlation Functions for I2EM Model.

    Parameters
    ----------
    n : int (>1)
        Coefficient needed for x-power and x-exponential
        correlation function
    wvnb : float
        Calculated by SurfScat Module.
    corrlength : int or float
        Correlation length (cm)
    sigma : int or float
        RMS Height (cm)
    Ts : float
        Calculated by SurfScat Module.
    """
    gauss = gaussian(n, wvnb, sigma, corrlength, Ts)
    exp = exponential(n, wvnb, sigma, corrlength, Ts)

    Wn = gauss.Wn / exp.Wn
    rss = gauss.rss / exp.rss

    return Wn, rss
