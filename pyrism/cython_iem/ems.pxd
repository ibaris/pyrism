# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""
cdef emsv_integralfunc(float x, float y, float iza, double complex eps, double complex rv, double complex rh, float k,
                      float kl, float ks,
                      double complex sq, corrfunc, float corrlength, int pol)

cdef calc_iem_ems(float iza, float k, float sigma, float corrlength, double complex eps, int corrfunc_ems)
cdef tuple calc_iem_ems_vec(double[:] iza, double[:] k, double[:] sigma, double[:] corrlength, double complex[:] eps,
                            int corrfunc_ems)
