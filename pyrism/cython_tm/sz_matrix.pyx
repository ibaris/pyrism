# -*- coding: utf-8 -*-
"""
Created on 28.02.2019 by Ismail Baris
"""
from __future__ import division

import numpy as np
cimport numpy as np

from pyrism.auxil.quadrature import get_points_and_weights

from pyrism.fortran_tm.fotm import calcampl

DTYPE = np.complex
ctypedef np.complex_t DTYPE_t

# ----------------------------------------------------------------------------------------------------------------------
# T-Matrix to Calculate S, Z
# ----------------------------------------------------------------------------------------------------------------------
# Single Orientation ---------------------------------------------------------------------------------------------------
# ---- NOT Vectorized ----
cdef tuple SZ_S(int nmax, double wavelength, double izaDeg, double vzaDeg, double iaaDeg,
                    double vaaDeg, double alphaDeg, double betaDeg):
    """
    Calculate the single scattering and phase matrix.
    
    Parameters
    ----------
    nmax : int
        Nmax parameter.
    wavelength : double
        Wavelength in same unit as radius (used by function calc_nmax).
    izaDeg, vzaDeg, iaaDeg, vaaDeg : double
        Incidence (iza) and scattering (vza) zenith angle and incidence and viewing
        azimuth angle (ira, vra) in [DEG].
    alphaDeg, betaDeg: double
        The Euler angles of the particle orientation in [DEG].

    Returns
    -------
    S, Z : tuple with double[:,:]
        Two dimensional scattering (S) and phase (Z) matrix.
    """

    cdef:
        np.ndarray[double, ndim=2]  Z
        np.ndarray[DTYPE_t, ndim=2]  S

    S = np.zeros((2, 2), dtype=np.complex)
    Z = np.zeros((4, 4), dtype=np.double)

    S, Z = calcampl(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)

    return S, Z

# ---- Vectorized ----
cdef tuple SZ_S_VEC(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] vzaDeg, double[:] iaaDeg,
                    double[:] vaaDeg, double[:] alphaDeg, double[:] betaDeg):
    """
    Calculate the single scattering and phase matrix in a vectorized function.
    
    Parameters
    ----------
    nmax : int[:] 
        Nmax parameter.
    wavelength : double[:]
        Wavelength in same unit as radius (used by function calc_nmax).
    izaDeg, vzaDeg, iaaDeg, vaaDeg : double[:]
        Incidence (iza) and scattering (vza) zenith angle and incidence and viewing
        azimuth angle (ira, vra) in [DEG].
    alphaDeg, betaDeg: double[:]
        The Euler angles of the particle orientation in [DEG].

    Returns
    -------
    S, Z : tuple with double[:,:,:]
        Three dimensional scattering (S) and phase (Z) matrix.
    """

    cdef:
        Py_ssize_t xmax = nmax.shape[0]
        Py_ssize_t x
        np.ndarray[double, ndim=3]  Z
        np.ndarray[DTYPE_t, ndim=3]  S

    S = np.zeros((xmax, 2, 2), dtype=np.complex)
    Z = np.zeros((xmax, 4, 4), dtype=np.double)


    for x in range(xmax):
        S[x], Z[x] = calcampl(nmax[x], wavelength[x], izaDeg[x], vzaDeg[x], iaaDeg[x],
                                        vaaDeg[x], alphaDeg[x], betaDeg[x])

    return S, Z

# Adaptive Average Fixed Orientation -----------------------------------------------------------------------------------
# NOT Vectorized
cdef tuple SZ_AF(int nmax, double wavelength, double izaDeg, double vzaDeg, double iaaDeg, double vaaDeg, int n_alpha,
                 int n_beta, or_pdf):
    """
    Calculate the variable orientation scattering and phase matrix.
    
    This method uses a fast Gaussian quadrature and is suitable
    for most use. Uses the set particle orientation PDF, ignoring
    the alpha and beta attributes.
    
    Parameters
    ----------
    nmax : int
        Nmax parameter.
    wavelength : double
        Wavelength in same unit as radius (used by function calc_nmax).
    izaDeg, vzaDeg, iaaDeg, vaaDeg : double
        Incidence (iza) and scattering (vza) zenith angle and incidence and viewing
        azimuth angle (ira, vra) in [DEG].
    n_alpha, n_beta : int
        Number of integration points in the alpha and beta Euler angle.
    or_pdf: callable
        Particle orientation Probability Density Function (PDF) for orientation averaging.
            
    Returns
    -------
    S, Z : tuple with double[:,:]
        Twoe dimensional scattering (S) and phase (Z) matrix.
    """
    cdef:
        double[:] beta_p, beta_w, alpha_array, wavelength_array, izaDeg_array, vzaDeg_array, iaaDeg_array, vaaDeg_array
        int[:] nmax_array
        np.ndarray[double, ndim=3]  Z, Z_temp
        np.ndarray[DTYPE_t, ndim=3]  S, S_temp
        np.ndarray[double, ndim=2]  Z_sum
        np.ndarray[DTYPE_t, ndim=2] S_sum
        Py_ssize_t x, y

        double aw = 1.0 / n_alpha
        double alpha_item
        double[:] alpha = np.linspace(0, 360, n_alpha + 1, dtype=np.double)[:-1]

    S_sum = np.zeros((2, 2), dtype=np.complex)
    Z_sum = np.zeros((4, 4))

    S = np.zeros((1, 2, 2), dtype=np.complex)
    Z = np.zeros((1, 4, 4))

    beta_p, beta_w = get_points_and_weights(or_pdf, 0, 180, num_points=n_beta)

    for item in alpha:
        alpha_array = np.zeros_like(beta_p) + item
        nmax_array = np.zeros_like(beta_p, dtype=np.intc) + np.asarray(nmax, dtype=np.intc).flatten()
        wavelength_array = np.zeros_like(beta_p) + np.asarray(wavelength).flatten()
        izaDeg_array = np.zeros_like(beta_p) + np.asarray(izaDeg).flatten()
        vzaDeg_array = np.zeros_like(beta_p) + np.asarray(vzaDeg).flatten()
        iaaDeg_array = np.zeros_like(beta_p) + np.asarray(iaaDeg).flatten()
        vaaDeg_array = np.zeros_like(beta_p) + np.asarray(vaaDeg).flatten()

        S_temp, Z_temp = SZ_S_VEC(nmax_array, wavelength_array, izaDeg_array, vzaDeg_array, iaaDeg_array,
                                  vaaDeg_array, alpha_array, beta_p)

        for y in range(S_temp.shape[0]):
            S_sum += beta_w[y] * S_temp[y]
            Z_sum += beta_w[y] * Z_temp[y]

        S[x] = S_sum
        Z[x] = Z_sum

    # normalize to get a proper average
    S *= aw / np.sum(beta_w)
    Z *= aw / np.sum(beta_w)

    return S[0], Z[0]

# ---- Vectorized ----
cdef tuple SZ_AF_VEC(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] vzaDeg, double[:] iaaDeg,
                     double[:] vaaDeg, int n_alpha, int n_beta, or_pdf):
    """
    Calculate the variable orientation scattering and phase matrix in a vectorized function.
    
    This method uses a fast Gaussian quadrature and is suitable
    for most use. Uses the set particle orientation PDF, ignoring
    the alpha and beta attributes.
    
    Parameters
    ----------
    nmax : int[:] 
        Nmax parameter.
    wavelength : double[:]
        Wavelength in same unit as radius (used by function calc_nmax).
    izaDeg, vzaDeg, iaaDeg, vaaDeg : double[:]
        Incidence (iza) and scattering (vza) zenith angle and incidence and viewing
        azimuth angle (ira, vra) in [DEG].
    n_alpha, n_beta : int
        Number of integration points in the alpha and beta Euler angle.
    or_pdf: callable
        Particle orientation Probability Density Function (PDF) for orientation averaging.
            
    Returns
    -------
    S, Z : tuple with double[:,:,:]
        Three dimensional scattering (S) and phase (Z) matrix.
    """
    cdef:
        double[:] beta_p, beta_w, alpha_array, wavelength_array, izaDeg_array, vzaDeg_array, iaaDeg_array, vaaDeg_array
        int[:] nmax_array
        np.ndarray[double, ndim=3]  Z, Z_temp
        np.ndarray[DTYPE_t, ndim=3]  S, S_temp
        np.ndarray[double, ndim=2]  Z_sum
        np.ndarray[DTYPE_t, ndim=2] S_sum
        Py_ssize_t x

        double aw = 1.0 / n_alpha
        double item
        double[:] alpha = np.linspace(0, 360, n_alpha + 1, dtype=np.double)[:-1]
        Py_ssize_t xmax = nmax.shape[0]

    S_sum = np.zeros((2, 2), dtype=np.complex)
    Z_sum = np.zeros((4, 4))

    S = np.zeros((xmax, 2, 2), dtype=np.complex)
    Z = np.zeros((xmax, 4, 4))

    beta_p, beta_w = get_points_and_weights(or_pdf, 0, 180, num_points=n_beta)

    for item in alpha:
        alpha_array = np.zeros_like(beta_p) + item
        for x in range(nmax.shape[0]):
            nmax_array = np.zeros_like(beta_p, dtype=np.intc) + np.asarray(nmax[x], dtype=np.intc).flatten()
            wavelength_array = np.zeros_like(beta_p) + np.asarray(wavelength[x]).flatten()
            izaDeg_array = np.zeros_like(beta_p) + np.asarray(izaDeg[x]).flatten()
            vzaDeg_array = np.zeros_like(beta_p) + np.asarray(vzaDeg[x]).flatten()
            iaaDeg_array = np.zeros_like(beta_p) + np.asarray(iaaDeg[x]).flatten()
            vaaDeg_array = np.zeros_like(beta_p) + np.asarray(vaaDeg[x]).flatten()

            S_temp, Z_temp = SZ_S_VEC(nmax_array, wavelength_array, izaDeg_array, vzaDeg_array, iaaDeg_array,
                                      vaaDeg_array, alpha_array, beta_p)

            for y in range(S_temp.shape[0]):
                S_sum += beta_w[y] * S_temp[y]
                Z_sum += beta_w[y] * Z_temp[y]

            S[x] = S_sum
            Z[x] = Z_sum

    # normalize to get a proper average
    S *= aw / np.sum(beta_w)
    Z *= aw / np.sum(beta_w)

    return S, Z
