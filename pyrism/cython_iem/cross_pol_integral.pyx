# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 03.03.2019 by Ismail Baris
"""
from __future__ import division
from libc.math cimport sin, cos, pow, sqrt

cdef xpol_integralfunc(double mui, double phi, char*sp, int n, double ks2, double cs, double s, double kl2,
                       double L, double complex er, double rss, double complex rvh, int n_spec):
    cdef:
        double cs2, r2, sf, csf, rx, ry, rp, rm, q, qt, a, b, c, d, B3, fvh1, fvh2, Fvh, au, fsh, sha
        Py_ssize_t nr = len()

    cs2 = pow(cs, 2)
    r2 = pow(mui, 2)
