# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 29.02.2019 by Ismail Baris
"""
from __future__ import division

import numpy as np
cimport numpy as np

cdef float PI = 3.14159265359


cdef double complex[:,:,:] M(double complex[:,:,:] S, double[:] wavelength, double[:] N):
    cdef:
        double[:] k
        # double complex[:] const
        double complex[:,:,:] MS
        Py_ssize_t o, p, q
        Py_ssize_t xmax = S.shape[0]

    # k = complex(0, (2. * PI) / wavelength)
    # const = complex(0, (2. * PI) / wavelength)
    # const = 1*1j*k

    MS = np.zeros_like(S)
    cdef double complex[:,:,:] MS_view = MS

    for o in range(xmax):
        for p in range(2):
            for q in range(2):
                MS_view[o, p, q] = N * complex(0, (2. * PI) / wavelength[o]) * S[o, p, q]

    return MS


cdef double[:,:,:] KE(double complex[:,:,:] S, double[:] wavelength, double[:] N):
    cdef:
        double[:,:,:] ke
        double[:] k
        double complex[:] const
        double complex[:,:,:] MS
        Py_ssize_t h, i, j
        Py_ssize_t xmax = S.shape[0]

    MS = M(S, wavelength, N)
    ke = np.zeros_like(S, dtype=np.double)
    cdef double[:,:,:] ke_view = ke

    for h in range(xmax):
        ke[h, 0, 0] = -MS[h, 0, 0].real
        ke[h, 0, 1] = 0.
        ke[h, 0, 2] = -MS[h, 0, 1].real
        ke[h, 0, 3] = -MS[h, 0, 1].imag

        ke[h, 1, 0] = 0.
        ke[h, 1, 1] = -MS[h, 1, 1].real
        ke[h, 1, 2] = -MS[h, 1, 0].real
        ke[h, 1, 3] = MS[h, 1, 0].imag

        ke[h, 2, 0] = -MS[h, 1, 0].real
        ke[h, 2, 1] = -MS[h, 0, 1].real
        ke[h, 2, 2] = -np.real(MS[h, 0, 0] + MS[h, 1, 1])
        ke[h, 2, 3] = np.imag(MS[h, 0, 0] - MS[h, 1, 1])

        ke[h, 2, 0] = 2*MS[h, 1, 0].imag
        ke[h, 2, 1] = -2*MS[h, 0, 1].imag
        ke[h, 2, 2] = -np.imag(MS[h, 0, 0] - MS[h, 1, 1])
        ke[h, 2, 3] = -np.real(MS[h, 0, 0] + MS[h, 1, 1])

    return ke

# cdef double[:,:,:] XSEC_QS_S_MAT(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg,
#                                double[:] alphaDeg, double[:] betaDeg, int Nx, int Ny, int verbose):
#     """Scattering Cross Section Matrix for single orientation.
#
#     Parameters
#     ----------
#     nmax : int[:]
#         Nmax parameter.
#     wavelength : double[:]
#         Wavelength in same unit as radius (used by function calc_nmax).
#     izaDeg, vzaDeg, iaaDeg, vaaDeg : double[:]
#         Incidence (iza) and scattering (vza) zenith angle and incidence and viewing
#         azimuth angle (ira, vra) in [DEG].
#     alphaDeg, betaDeg: double[:]
#         The Euler angles of the particle orientation in [DEG].
#
#     Returns
#     -------
#     Qs : double[:,:]
#         Two dimensional scattering cross section.
#     """
#     cdef:
#         np.ndarray xlin, ylin, Xf, Yf, Zf, I, memorize, sin_Xf, X, Y, Zx, vzaDeg, vaaDeg, Zfx, cos_t0, sin_t0, multi
#         double[:,:,:] Qs
#         Py_ssize_t x, j, k
#         Py_ssize_t xmax = nmax.shape[0]
#
#     xlin = np.linspace(0, PI / 2, Nx)
#     ylin = np.linspace(0, 2 * PI, Ny)
#     X, Y = np.meshgrid(xlin, ylin)
#
#     Xf = X.flatten()
#     Yf = Y.flatten()
#
#     vzaDeg = Xf * RAD_TO_DEG
#     vaaDeg = Yf * RAD_TO_DEG
#
#     multi = sin(vzaDeg)
#
#     S, Z = SZ_S_VEC(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)
#
#     QS = np.zeros((xmax, 4, 4), dtype=np.double)
#     cdef double[:,:] QS_view = QS
#
#     for k in range(Z.shape[0]):
#         Zf = Z[k].flatten()
#         memorize = np.zeros_like(Zf)
#
#         for x in range(8):
#             Zfx = Zf[x] * sin(Xf)
#
#             Zx = Zfx.reshape((Nx, Ny))
#
#             I = np.zeros(Ny)
#
#             for j in range(Ny):
#                 I[j] = np.trapz(Zx[j, :], xlin)
#
#             memorize[x] = np.trapz(I, ylin)
#
#         QS_view[k] = memorize.reshape((4, 4))
#
#     return Qs
