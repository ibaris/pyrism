# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""
from __future__ import division
import numpy as np
cimport numpy as np
from scipy.integrate import dblquad
from libc.math cimport sin

from pyrism.cython_iem.auxil cimport (real_RaV_integration_ifunc, real_RaH_integration_ifunc,
                                      imag_RaV_integration_ifunc, imag_RaH_integration_ifunc)
from pyrism.cython_iem.fresnel cimport  compute_rt, compute_Rx0, compute_Rxi

cdef:
    double PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164

cdef tuple Rax_integration(double[:] iza, double[:] sigma, double[:] corrlength, double complex[:] eps):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double bound, ravv, rahh
        double complex[:] Rav_view, Rah_view
        double sigy, sigx

    Rav = np.zeros_like(iza, dtype=np.complex)
    Rah = np.zeros_like(iza, dtype=np.complex)

    Rav_view = Rav
    Rah_view = Rah

    for i in range(xmax):
        sigx = 1.1 * sigma[i] / corrlength[i]
        sigy = sigx

        bound = 3 * sigx

        ravv_real = dblquad(real_RaV_integration_ifunc, -bound, bound, lambda x: -bound, lambda x: bound,
                   args=(iza[i], sigx, sigy, eps[i]))[0]

        rahh_real = dblquad(real_RaH_integration_ifunc, -bound, bound, lambda x: -bound, lambda x: bound,
                   args=(iza[i], sigx, sigy, eps[i]))[0]

        ravv_imag = dblquad(imag_RaV_integration_ifunc, -bound, bound, lambda x: -bound, lambda x: bound,
                   args=(iza[i], sigx, sigy, eps[i]))[0]

        rahh_imag = dblquad(imag_RaH_integration_ifunc, -bound, bound, lambda x: -bound, lambda x: bound,
                   args=(iza[i], sigx, sigy, eps[i]))[0]


        # ravv = dblquad(RaV_integration_ifunc, -bound, bound, lambda x: -bound, lambda x: bound,
        #            args=(iza[i], sigx, sigy, eps[i]))[0]
        #
        # rahh = dblquad(RaH_integration_ifunc, -bound, bound, lambda x: -bound, lambda x: bound,
        #            args=(iza[i], sigx, sigy, eps[i]))[0]

        Rav_view[i] = complex(ravv_real, ravv_imag) / (2 * PI * sigx * sigy)
        Rah_view[i] = complex(rahh_real, rahh_imag) / (2 * PI * sigx * sigy)

    return Rav, Rah

cdef tuple compute_Rxt(double[:] iza, double[:] vza, double[:] raa, double[:] sigma, double[:] corrlength,
                       double complex[:] eps, double[:] Tf):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double complex[:] RaV, RaH
        double complex[:] Rvt_view, Rht_view, rt, Rv0, Rh0, Rvi, Rhi

    rt = compute_rt(iza, eps.base.real, eps.base.imag)

    Rv0, Rh0 = compute_Rx0(eps)
    Rvi, Rhi = compute_Rxi(iza, eps, rt)

    Rvt = np.zeros_like(iza, dtype=np.complex)
    Rht = np.zeros_like(iza, dtype=np.complex)

    RaV, RaH = Rax_integration(iza, sigma, corrlength, eps)

    Rvt_view = Rvt
    Rht_view = Rht

    for i in range(xmax):
        if vza[i] == iza[i] and np.allclose(raa[i], PI):
            Rvt_view[i] = Rvi[i] + (Rv0[i] - Rvi[i]) * Tf[i]
            Rht_view[i] = Rhi[i] + (Rh0[i] - Rhi[i]) * Tf[i]

        else:
            Rvt_view[i] = RaV[i]
            Rht_view[i] = RaH[i]

    return Rvt, Rht

# Compute fvv and fhh ----------------------------------------------------------------------------------------------
cdef tuple compute_fxx(double complex[:] Rvt, double complex[:] Rht, double[:] iza, double[:] vza, double[:] raa):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double temp
        double complex[:] fvv_view, fhh_view

    fvv = np.zeros_like(Rvt, dtype=np.complex)
    fhh = np.zeros_like(Rht, dtype=np.complex)

    fvv_view = fvv
    fhh_view = fhh

    for i in range(xmax):
        temp = ((sin(iza[i]) * sin(vza[i]) - (1 + np.cos(iza[i]) * np.cos(vza[i])) *
                np.cos(raa[i])) / (np.cos(iza[i]) + np.cos(vza[i])))

        fvv_view[i] = 2 * Rvt[i] * temp
        fhh_view[i] = -2 * Rht[i] * temp

    return fvv, fhh
