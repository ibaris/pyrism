# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""
from __future__ import division
import numpy as np
cimport numpy as np
from scipy.integrate import dblquad
from libc.math cimport sin, cos
from pyrism.cython_iem.auxil cimport factorial
DTYPE = np.float

ctypedef np.float_t DTYPE_t

cdef:
    double PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164

cdef emsv_integralfunc(float x, float y, float iza, double complex eps, double complex rv, double complex rh, float k,
                      float kl, float ks,
                      double complex sq, corrfunc, float corrlength, int pol):
    cdef int nr
    cdef np.ndarray wn, expwn, gauswn, svv, shh
    cdef float wnn, vv, hv, ref, hh, vh
    cdef double complex Fvv, Fhv, Fvvs, Fhvs, Ivv, Ihv, Ihh, Ivh

    cdef float error = 1.0e3
    cdef double complex sqs = np.sqrt(eps - sin(x) ** 2)
    cdef double complex rc = (rv - rh) / 2
    cdef double complex tv = 1 + rv
    cdef double complex th = 1 + rh

    # -- calc coefficients for surface correlation spectra
    cdef float wvnb = k * np.sqrt(sin(iza) ** 2 - 2 * sin(iza) * sin(x) * cos(y) + sin(x) ** 2)

    try:
        nr = len(x)
    except (IndexError, TypeError):
        nr = 1

    # -- calculate number of spectral components needed
    cdef int n_spec = 1
    while error > 1.0e-3:
        n_spec = n_spec + 1
        #   error = (ks2 *(cs + css)**2 )**n_spec / factorial(n_spec)
        # ---- in this case we will use the smallest ths to determine the number of
        # spectral components to use.  It might be more than needed for other angles
        # but this is fine.  This option is used to simplify calculations.
        error = (ks ** 2 * (cos(iza) + cos(x)) ** 2) ** n_spec / factorial(n_spec)
        error = np.min(error)
    # -- calculate expressions for the surface spectra

    if corrfunc == 1:
        wn = np.zeros([n_spec, nr])

        for n in range(n_spec):
            wn[n, :] = (n + 1) * kl ** 2 / ((n + 1) ** 2 + (wvnb * corrlength) ** 2) ** 1.5

    if corrfunc == 2:
        wn = np.zeros([n_spec, nr])

        for n in range(n_spec):
            wn[n, :] = 0.5 * kl ** 2 / (n + 1) * np.exp(-(wvnb * corrlength) ** 2 / (4 * (n + 1)))

    if corrfunc == 3:
        expwn = np.zeros([n_spec, nr])
        gauswn = np.zeros([n_spec, nr])

        for n in range(n_spec):
            expwn[n, :] = (n + 1) * kl ** 2 / ((n + 1) ** 2 + (wvnb * corrlength) ** 2) ** 1.5
            gauswn[n, :] = 0.5 * kl ** 2 / (n + 1) * np.exp(-(wvnb * corrlength) ** 2 / (4 * (n + 1)))

        wn = expwn / gauswn

    # -- calculate fpq!

    cdef float ff = 2 * (sin(iza) * sin(x) - (1 + cos(iza) * cos(x)) * cos(y)) / (
            cos(iza) + cos(x))

    cdef double complex fvv = rv * ff
    cdef double complex fhh = -rh * ff

    cdef double complex fvh = -2 * rc * sin(y)
    # cdef double complex fhv = 2 * rc * sin(y)

    # -- calculate Fpq and Fpqs -----
    cdef double complex fhv = sin(iza) * (sin(x) - sin(iza) * cos(y)) / (cos(iza) ** 2 * cos(x))
    cdef double complex T = (sq * (cos(iza) + sq) + cos(iza) * (eps * cos(iza) + sq)) / (
            eps * cos(iza) * (cos(iza) + sq) + sq * (eps * cos(iza) + sq))

    cdef double complex cm2 = cos(x) * sq / cos(iza) / sqs - 1
    cdef float ex = np.exp(-ks ** 2 * cos(iza) * cos(x))
    cdef float de = 0.5 * np.exp(-ks ** 2 * (cos(iza) ** 2 + cos(x) ** 2))

    if pol == 1:
        Fvv = (eps - 1) * sin(iza) ** 2 * tv ** 2 * fhv / eps ** 2
        Fhv = (T * sin(iza) * sin(iza) - 1. + cos(iza) / cos(x) + (
                eps * T * cos(iza) * cos(x) * (
                eps * T - sin(iza) * sin(iza)) - sq * sq) / (
                       T * eps * sq * cos(x))) * (1 - rc * rc) * sin(y)

        Fvvs = -cm2 * sq * tv ** 2 * (
                cos(y) - sin(iza) * sin(x)) / cos(
            iza) ** 2 / eps - cm2 * sqs * tv ** 2 * cos(y) / eps - (
                       cos(x) * sq / cos(iza) / sqs / eps - 1) * sin(
            x) * tv ** 2 * (
                       sin(iza) - sin(x) * cos(y)) / cos(iza)
        Fhvs = -(sin(x) * sin(x) / T - 1 + cos(x) / cos(iza) + (
                cos(iza) * cos(x) * (
                1 - sin(x) * sin(x) * T) - T * T * sqs * sqs) / (
                         T * sqs * cos(iza))) * (1 - rc * rc) * sin(y)

        # -- calculate the bistatic field coefficients ---

        svv = np.zeros([n_spec, nr])
        for n in range(n_spec):
            Ivv = fvv * ex * (ks * (cos(iza) + cos(x))) ** (n + 1) + (
                    Fvv * (ks * cos(x)) ** (n + 1) + Fvvs * (ks * cos(iza)) ** (n + 1)) / 2
            Ihv = fhv * ex * (ks * (cos(iza) + cos(x))) ** (n + 1) + (
                    Fhv * (ks * cos(x)) ** (n + 1) + Fhvs * (ks * cos(iza)) ** (n + 1)) / 2

        wnn = wn[n, :] / factorial(n + 1)
        vv = wnn * (abs(Ivv)) ** 2
        hv = wnn * (abs(Ihv)) ** 2
        svv[n, :] = (de * (vv + hv) * sin(x) * (1 / cos(iza))) / (4 * PI)

        ref = np.sum([svv])  # adding all n terms stores in different rows

    if pol == 2:
        Fhh = -(eps - 1) * th ** 2 * fhv
        Fvh = (sin(iza) * sin(iza) / T - 1. + cos(iza) / cos(x) + (
                cos(iza) * cos(x) * (
                1 - sin(iza) * sin(iza) * T) - T * T * sq * sq) / (
                       T * sq * cos(x))) * (1 - rc * rc) * sin(y)

        Fhhs = cm2 * sq * th ** 2 * (
                cos(y) - sin(iza) * sin(x)) / cos(
            iza) ** 2 + cm2 * sqs * th ** 2 * cos(y) + cm2 * sin(x) * th ** 2 * (
                       sin(iza) - sin(x) * cos(y)) / cos(iza)
        Fvhs = -(T * sin(x) * sin(x) - 1 + cos(x) / cos(iza) + (
                eps * T * cos(iza) * cos(x) * (
                eps * T - sin(x) * sin(x)) - sqs * sqs) / (
                         T * eps * sqs * cos(iza))) * (1 - rc * rc) * sin(y)

        shh = np.zeros([n_spec, nr])
        for n in range(n_spec):
            Ihh = fhh * ex * (ks * (cos(iza) + cos(x))) ** (n + 1) + (
                    Fhh * (ks * cos(x)) ** (n + 1) + Fhhs * (ks * cos(iza)) ** (n + 1)) / 2
            Ivh = fvh * ex * (ks * (cos(iza) + cos(x))) ** (n + 1) + (
                    Fvh * (ks * cos(x)) ** (n + 1) + Fvhs * (ks * cos(iza)) ** (n + 1)) / 2

        wnn = wn[n, :] / factorial(n + 1)
        hh = wnn * (abs(Ihh)) ** 2
        vh = wnn * (abs(Ivh)) ** 2
        (2 * (3 + 4) * sin(5) * 1 / cos(6)) / (PI * 4)
        shh[n, :] = (de * (hh + vh) * sin(x) * (1 / cos(iza))) / (4 * PI)

        ref = np.sum([shh])

    return ref

cdef calc_iem_ems(float iza, float k, float sigma, float corrlength, double complex eps, int corrfunc_ems):
    cdef float ks = k * sigma
    cdef float kl = k * corrlength

    # -- calculation of reflection coefficients
    cdef double complex sq = np.sqrt(eps - np.sin(iza) ** 2)

    cdef double complex rv = (eps * np.cos(iza) - sq) / (
            eps * np.cos(iza) + sq)

    cdef double complex rh = (np.cos(iza) - sq) / (np.cos(iza) + sq)

    cdef float refv = dblquad(emsv_integralfunc, 0, PI / 2, lambda x: 0, lambda x: PI,
                              args=(iza, eps, rv, rh, k, kl, ks, sq, corrfunc_ems, corrlength, 1))[0]

    cdef float refh = dblquad(emsv_integralfunc, 0, PI / 2, lambda x: 0, lambda x: PI,
                              args=(iza, eps, rv, rh, k, kl, ks, sq, corrfunc_ems, corrlength, 2))[0]

    cdef float VV = 1 - refv - np.exp(-ks ** 2 * np.cos(iza) * np.cos(iza)) * (
        abs(rv)) ** 2
    cdef float HH = 1 - refh - np.exp(-ks ** 2 * np.cos(iza) * np.cos(iza)) * (
        abs(rh)) ** 2

    return VV, HH

cdef tuple calc_iem_ems_vec(double[:] iza, double[:] k, double[:] sigma, double[:] corrlength, double complex[:] eps,
                            int corrfunc_ems):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t x
        double VV_temp, HH_temp
        double[:] VV_view, HH_view

    VV = np.zeros_like(iza)
    HH = np.zeros_like(iza)
    VV_view = VV
    HH_view = HH

    for x in range(xmax):
        VV_temp, HH_temp = calc_iem_ems(iza[x], k[x], sigma[x], corrlength[x], eps[x], corrfunc_ems)
        VV_view[x] = VV_temp
        HH_view[x] = HH_temp

    return VV, HH
