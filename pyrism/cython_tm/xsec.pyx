# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 29.02.2019 by Ismail Baris
"""
from __future__ import division

import numpy as np
cimport numpy as np
from pyrism.cython_tm.sz_matrix cimport SZ_S_VEC, SZ_AF_VEC

cdef :
    float PI = 3.14159265359
    float DEG_TO_RAD = PI / 180.0
    float RAD_TO_DEG = 180.0 / PI


cdef double complex[:,:,:] M(double complex[:,:,:] S, double[:] wavelength):
    cdef:
        double[:] k
        # double complex[:] const
        double complex[:,:,:] MS
        Py_ssize_t o, p, q
        Py_ssize_t xmax = S.shape[0]

    MS = np.zeros_like(S)
    cdef double complex[:,:,:] MS_view = MS

    for o in range(xmax):
        for p in range(2):
            for q in range(2):
                MS_view[o, p, q] = wavelength[o] * S[o, p, q]

    return MS

cdef double[:,:,:] XE(double complex[:,:,:] S, double[:] wavelength):
    cdef:
        double[:,:,:] ke
        double[:] k
        double complex[:] const
        double complex[:,:,:] MS
        Py_ssize_t h, i, j
        Py_ssize_t xmax = S.shape[0]

    MS = M(S, wavelength)
    ke = np.zeros((xmax, 4, 4), dtype=np.double)
    cdef double[:,:,:] ke_view = ke

    for h in range(xmax):
        ke_view[h, 0, 0] = ke_view[h, 1, 1] = ke_view[h, 2, 2] = ke_view[h, 3, 3] = MS[h, 0, 0].imag + MS[h, 1, 1].imag
        ke_view[h, 0, 1] = ke_view[h, 1, 0] = MS[h, 0, 0].imag - MS[h, 1, 1].imag
        ke_view[h, 0, 2] = ke_view[h, 2, 0] = MS[h, 0, 1].imag + MS[h, 1, 0].imag
        ke_view[h, 0, 3] = ke_view[h, 3, 0] = -MS[h, 0, 1].imag + MS[h, 1, 0].imag

        ke_view[h, 1, 2] = MS[h, 0, 1].imag - MS[h, 1, 0].imag
        ke_view[h, 1, 3] = -(MS[h, 0, 1].real + MS[h, 1, 0].real)

        ke_view[h, 2, 1] = -ke_view[h, 1, 2]
        ke_view[h, 3, 1] = -ke_view[h, 1, 3]

        ke_view[h, 2, 3] = MS[h, 1, 1].real - MS[h, 0, 0].real
        ke_view[h, 3, 2] = -ke_view[h, 2, 3]

    return ke

cdef double[:,:] XSEC_QSI(double[:,:,:] Z):
    """Intensity.

    Parameters
    ----------
    Z : double complex[:,:,:]
        Three dimensional phase matrix.

    Returns
    -------
    Qi : double[:,:]
        Two dimensional intensity.
    """
    cdef:
        np.ndarray[double, ndim=2]  QSI
        Py_ssize_t x
        Py_ssize_t xmax = Z.shape[0]

    QSI = np.zeros((xmax, 2), dtype=np.double)

    for x in range(xmax):
        QSI[x, 0] = Z[x, 0, 0] + Z[x, 0, 1]
        QSI[x, 1] = Z[x, 0, 0] - Z[x, 0, 1]

    return QSI

cdef double[:,:] XS_S(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg,
                          double[:] alphaDeg, double[:] betaDeg, int Nx, int Ny, int verbose, int asy):

    cdef:
        np.ndarray xlin, ylin, X, Y
        double[:] Xf, Yf, Zx, I_view, vzaDeg_view, vaaDeg_view, Qs_view
        double F
        Py_ssize_t x, j, k
        list QSf

    xlin = np.linspace(0, PI, Nx)
    ylin = np.linspace(0, 2 * PI, Ny)
    X, Y = np.meshgrid(xlin, ylin)

    Xf = X.flatten()
    Yf = Y.flatten()

    vzaDeg = np.zeros_like(Xf)
    vaaDeg = np.zeros_like(Yf)

    vzaDeg_view = vzaDeg
    vaaDeg_view = vaaDeg

    for x in range(Xf.shape[0]):
        vzaDeg_view[x] = Xf[x] * RAD_TO_DEG
        vaaDeg_view[x] = Yf[x] * RAD_TO_DEG

    S, Z = SZ_S_VEC(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)

    QSf = list()

    if asy:
        cos_t0 = np.cos(izaDeg.base * DEG_TO_RAD)
        sin_t0 = np.sin(izaDeg.base * DEG_TO_RAD)

        multi = 0.5 * (np.sin(2 * Xf.base) * cos_t0 +
                       (1 - np.cos(2 * Xf.base)) * sin_t0 * np.cos((iaaDeg.base * DEG_TO_RAD) - Yf.base))

    else:
        multi = np.sin(Xf)

    for j in range(Z.shape[1]):
        for k in range(Z.shape[2]):
            Zx = Z[:, j, k] * multi

            Y = Zx.base.reshape((Nx, Ny))

            I = np.zeros(Ny)
            I_view = I

            for l in range(Ny):
                I_view[l] = np.trapz(Y[l, :], xlin)

            F = np.trapz(I, ylin)

            QSf.append(F)

    Qs = np.zeros(16)
    Qs_view = Qs

    for x in range(16):
        Qs_view[x] = QSf[x]

    Qsr = Qs_view.base.reshape((4, 4))

    return Qsr

cdef double[:,:] XS_AF(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg, int Nx, int Ny,
                       int n_alpha, int n_beta, or_pdf, int verbose, int asy):

    cdef:
        np.ndarray xlin, ylin, X, Y
        double[:] Xf, Yf, Zx, I_view, vzaDeg_view, vaaDeg_view, Qs_view
        double F
        Py_ssize_t x, j, k
        list QSf

    xlin = np.linspace(0, PI, Nx)
    ylin = np.linspace(0, 2 * PI, Ny)
    X, Y = np.meshgrid(xlin, ylin)

    Xf = X.flatten()
    Yf = Y.flatten()

    vzaDeg = np.zeros_like(Xf)
    vaaDeg = np.zeros_like(Yf)

    vzaDeg_view = vzaDeg
    vaaDeg_view = vaaDeg

    for x in range(Xf.shape[0]):
        vzaDeg_view[x] = Xf[x] * RAD_TO_DEG
        vaaDeg_view[x] = Yf[x] * RAD_TO_DEG

    S, Z = SZ_AF_VEC(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, n_alpha, n_beta, or_pdf)

    QSf = list()

    if asy:
        cos_t0 = np.cos(izaDeg.base * DEG_TO_RAD)
        sin_t0 = np.sin(izaDeg.base * DEG_TO_RAD)

        multi = 0.5 * (np.sin(2 * Xf.base) * cos_t0 +
                       (1 - np.cos(2 * Xf.base)) * sin_t0 * np.cos((iaaDeg.base * DEG_TO_RAD) - Yf.base))
    else:
        multi = np.sin(Xf)

    for j in range(Z.shape[1]):
        for k in range(Z.shape[2]):
            Zx = Z[:, j, k] * multi

            Y = Zx.base.reshape((Nx, Ny))

            I = np.zeros(Ny)
            I_view = I

            for l in range(Ny):
                I_view[l] = np.trapz(Y[l, :], xlin)

            F = np.trapz(I, ylin)

            QSf.append(F)

    Qs = np.zeros(16)
    Qs_view = Qs

    for x in range(16):
        Qs_view[x] = QSf[x]

    Qsr = Qs_view.base.reshape((4, 4))

    return Qsr

# ------------------------------------------------------------------------------------------------------------
# Port this to SPINPY
# ------------------------------------------------------------------------------------------------------------
# cdef double complex[:,:,:] M(double complex[:,:,:] S, double[:] wavelength, double[:] N):
#     cdef:
#         double[:] k
#         # double complex[:] const
#         double complex[:,:,:] MS
#         Py_ssize_t o, p, q
#         Py_ssize_t xmax = S.shape[0]
#
#     # k = complex(0, (2. * PI) / wavelength)
#     # const = complex(0, (2. * PI) / wavelength)
#     # const = 1*1j*k
#
#     MS = np.zeros_like(S)
#     cdef double complex[:,:,:] MS_view = MS
#
#     for o in range(xmax):
#         for p in range(2):
#             for q in range(2):
#                 MS_view[o, p, q] = N[0] * wavelength[o] * S[o, p, q]
#
#     return MS
#
#
# cdef double[:,:,:] KE(double complex[:,:,:] S, double[:] wavelength, double[:] N):
#     cdef:
#         double[:,:,:] ke
#         double[:] k
#         double complex[:] const
#         double complex[:,:,:] MS
#         Py_ssize_t h, i, j
#         Py_ssize_t xmax = S.shape[0]
#
#     MS = M(S, wavelength, N)
#     ke = np.zeros((xmax, 4, 4), dtype=np.double)
#     cdef double[:,:,:] ke_view = ke
#
#     for h in range(xmax):
#         ke_view[h, 0, 0] = -MS[h, 0, 0].real
#         ke_view[h, 0, 1] = 0.
#         ke_view[h, 0, 2] = -MS[h, 0, 1].real
#         ke_view[h, 0, 3] = -MS[h, 0, 1].imag
#
#         ke_view[h, 1, 0] = 0.
#         ke_view[h, 1, 1] = -MS[h, 1, 1].real
#         ke_view[h, 1, 2] = -MS[h, 1, 0].real
#         ke_view[h, 1, 3] = MS[h, 1, 0].imag
#
#         ke_view[h, 2, 0] = -MS[h, 1, 0].real
#         ke_view[h, 2, 1] = -MS[h, 0, 1].real
#         ke_view[h, 2, 2] = -np.real(MS[h, 0, 0] + MS[h, 1, 1])
#         ke_view[h, 2, 3] = np.imag(MS[h, 0, 0] - MS[h, 1, 1])
#
#         ke_view[h, 3, 0] = 2.*MS[h, 1, 0].imag
#         ke_view[h, 3, 1] = -2.*MS[h, 0, 1].imag
#         ke_view[h, 3, 2] = -np.imag(MS[h, 0, 0] - MS[h, 1, 1])
#         ke_view[h, 3, 3] = -np.real(MS[h, 0, 0] + MS[h, 1, 1])
#
#     return ke
