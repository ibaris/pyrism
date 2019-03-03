# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""
from __future__ import division
from libc.math cimport sin, cos, pow, sqrt
from pyrism.cython_iem.fresnel cimport  compute_rt, compute_Rxi
from pyrism.cython_iem.rspectrum cimport compute_TS_X
from pyrism.cython_iem.auxil cimport compute_ShdwS_X

cdef iem_x(double[:] k, double[:] iza, double complex[:] eps, double[:] corrlength,
           double[:] sigma, int[:] n, char* corrfunc):
    cdef:
        double[:] ks, kl, cs, s, rss, ShdwS
        double complex[:] rt, rv, rh
        int [:] n_spec

    ks = k * sigma
    kl = k * corrlength

    cs = cos(iza)
    s = sin(iza)

    rt = compute_rt(iza=iza, epsr=eps.base.real, epsi=eps.base.imag)
    rv, rh = compute_Rxi(iza=iza, eps=eps, rt=rt)
    rvh = (rv - rh) / 2

    if corrfunc == 'exponential':
        rss = sigma / corrlength
    elif corrfunc == ' gaussian':
        rss = (sigma / corrlength) * sqrt(2)
    elif corrfunc == 'xpower':
        rss = (sigma / corrlength) * sqrt(2*n)

    n_spec = compute_TS_X(iza, sigma, k)
    ShdwS = compute_ShdwS_X(iza=iza, rss=rss)
