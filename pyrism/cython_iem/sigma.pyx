# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""
from __future__ import division
import numpy as np
cimport numpy as np
from pyrism.cython_iem.auxil cimport factorial

cdef tuple compute_sigma_nought(int[:] Ts, double[:, :] Wn, double complex[:, :] Ivv, double complex[:, :] Ihh,
                                double[:] ShdwS, double[:] k, double[:] kz_iza, double[:] kz_vza, double[:] sigma):
    cdef:
        Py_ssize_t xmax = kz_iza.shape[0]
        Py_ssize_t i, index, i_index

        double sigmavv, sigmahh, a0, sigma2i, kz_xzai2, fact
        double[:] VV_view, HH_view

    VV = np.zeros(xmax, dtype=np.double)
    HH = np.zeros(xmax, dtype=np.double)

    VV_view = VV
    HH_view = HH

    sigmavv, sigmahh = 0.0, 0.0

    for i in range(xmax):
        sigma2i = pow(sigma[i], 2)
        kz_xzai2 = pow(kz_iza[i], 2) + pow(kz_vza[i], 2)

        for j in range(1, Ts[i] + 1):
            index = j - 1

            a0 = Wn[i, index] / factorial(j) * pow(sigma[i], (2 * j))

            sigmavv += pow(abs(Ivv[i, index]), 2) * a0
            sigmahh += pow(abs(Ihh[i, index]), 2) * a0

        VV_view[i] = sigmavv * ShdwS[i] * pow(k[i], 2) / 2 * np.exp(-sigma2i * kz_xzai2)
        HH_view[i] = sigmahh * ShdwS[i] * pow(k[i], 2) / 2 * np.exp(-sigma2i * kz_xzai2)

    return VV, HH
