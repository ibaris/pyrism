# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.01.2019 by Ismail Baris
"""
from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, pow, sqrt
from pyrism.cython_iem.auxil cimport factorial

cdef double[:] compute_wvnb(double[:] iza, double[:] vza, double[:] raa, double[:] phi, double[:] k):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double[:] wvnb_view
        double sin_vza, sin_iza, cos_raa, cos_phi, sin_raa, sin_phi, part1, part2


    wvnb = np.zeros_like(iza, dtype=np.double)
    wvnb_view = wvnb

    for i in range(xmax):
        sin_vza = sin(vza[i])
        cos_raa = cos(raa[i])
        sin_iza = sin(iza[i])
        cos_phi = cos(phi[i])
        sin_raa = sin(raa[i])
        sin_phi = sin(phi[i])

        part1 = pow((sin_vza * cos_raa - sin_iza * cos_phi), 2)
        part2 = pow((sin_vza * sin_raa - sin_iza * sin_phi), 2)

        wvnb_view[i] = k[i] * sqrt(part1 + part2)

    return wvnb

cdef int[:] compute_TS(double[:] iza, double[:] vza, double[:] sigma, double[:] k):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        int[:] TS_view
        int TS_temp
        double mui, muv, merror, error

    TS = np.zeros_like(iza, dtype=np.intc)
    TS_view = TS

    for i in range(xmax):
        mui = cos(iza[i])
        muv = cos(vza[i])
        TS_temp = 1
        error = 1.0

        while error > 1.0e-8: # and TS_temp <= 150:
            TS_temp += 1
            error = pow(pow(k[i] * sigma[i], 2) * pow((mui + muv), 2), TS_temp) / factorial(TS_temp)

        TS_view[i] = TS_temp

    return TS

cdef tuple compute_Wn_rss(corrfunc, double[:] iza, double[:] vza, double[:] raa, double[:] phi, double[:] k,
                          double[:] sigma, double[:] corrlength, int[:] n):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i, j

        double rss_temp
        double[:, :] Wn_view
        double[:] rss_view, wvnb, Wn_temp
        int[:] Ts

    wvnb = compute_wvnb(iza=iza, vza=vza, raa=raa, phi=phi, k=k)
    Ts = compute_TS(iza=iza, vza=vza, sigma=sigma, k=k)

    Wn = np.zeros((xmax, max(Ts.base)))
    rss = np.zeros_like(iza)

    Wn_view = Wn
    rss_view = rss

    for i in range(xmax):
        Wn_temp, rss_temp = corrfunc(sigma[i], corrlength[i], wvnb[i], Ts[i])

        for j in range(Wn_temp.shape[0]):
            Wn_view[i, j] = Wn_temp[j]

        rss_view[i] = rss_temp

    return Wn, rss

cdef int[:] compute_TS_X(double[:] iza, double[:] sigma, double[:] k):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        int[:] TS_view
        int TS_temp
        double mui, merror, error

    TS = np.zeros_like(iza, dtype=np.intc)
    TS_view = TS

    for i in range(xmax):
        mui = cos(iza[i])
        TS_temp = 1
        error = 1.0

        while error > 1.0e-8: # and TS_temp <= 150:
            TS_temp += 1
            error = pow(pow(k[i] * sigma[i], 2) * pow((2*mui), 2), TS_temp) / factorial(TS_temp)

        TS_view[i] = TS_temp

    return TS
