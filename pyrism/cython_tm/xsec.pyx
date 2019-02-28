# -*- coding: utf-8 -*-
"""
Created on 28.02.2019 by Ismail Baris
"""
from __future__ import division
import numpy as np
cimport numpy as np

cdef double[:,:] XSEC_QE(double complex[:,:,:] S, double[:] wavelength):
    """Extinction Matrix.

    Parameters
    ----------
    S : double complex[:,:,:]
        Three dimensoinal scattering matrix.
    wavelength : double[:]
        Wavelength in same unit as radius (used by function calc_nmax).
    N : double[:]
    
    Returns
    -------
    Qe : double[:,:]
        Two dimensional extinction cross section.
    """
    cdef:
        np.ndarray[double, ndim=2]  Qe
        Py_ssize_t x
        Py_ssize_t xmax = S.shape[0]

    QE = np.zeros((xmax, 2), dtype=np.double)
    cdef double[:,:] QE_view = QE

    for x in range(xmax):
        QE_view[x, 0] = 2 * wavelength[x] * S[x][0, 0].imag
        QE_view[x, 1] = 2 * wavelength[x] * S[x][1, 1].imag

    return QE
