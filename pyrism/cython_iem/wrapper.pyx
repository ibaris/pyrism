# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""
from __future__ import division
from pyrism.cython_iem.i2em cimport compute_i2em, compute_ixx
from pyrism.cython_iem.ems cimport calc_iem_ems_vec

def i2em(double[:] k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                 double[:] phi, double complex[:] eps, double[:] corrlength, double[:] sigma, corrfunc, int[:] n):

    return compute_i2em(k=k, kz_iza=kz_iza, kz_vza=kz_vza, iza=iza, vza=vza, raa=raa, phi=phi, eps=eps,
                        corrlength=corrlength, sigma=sigma, corrfunc=corrfunc, n=n)

def ixx(double[:] k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                 double[:] phi, double complex[:] eps, double[:] corrlength, double[:] sigma, corrfunc, int[:] n):

    return compute_ixx(k=k, kz_iza=kz_iza, kz_vza=kz_vza, iza=iza, vza=vza, raa=raa, phi=phi, eps=eps,
                        corrlength=corrlength, sigma=sigma, corrfunc=corrfunc, n=n)

def i2em_ems(double[:] iza, double[:] k, double[:] sigma, double[:] corrlength, double complex[:] eps,
             int corrfunc_ems):
    return calc_iem_ems_vec(iza, k, sigma, corrlength, eps, corrfunc_ems)
