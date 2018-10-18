# cython: cdivision=True
from __future__ import division

cimport numpy as np
import numpy as np
from pyrism.auxil.quadrature import get_points_and_weights

DTYPE = np.float
from libc.math cimport sin

ctypedef np.float_t DTYPE_t
from scipy.integrate import dblquad
from pyrism.fortran_tm import fotm as tmatrix

PI = 3.14159265359

DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI

# ----------------------------------------------------------------------------------------------------------------------
# T-Matrix to calculate nmax, S and Z
# ----------------------------------------------------------------------------------------------------------------------
# ---- Auxiliary functions ----
# Integration functions
cdef ifunc_S(float betaDeg, float alphaDeg, int i, int j, int real, float nmax, float wavelength, float izaDeg,
             float vzaDeg, float iaaDeg, float vaaDeg, or_pdf):
    cdef np.ndarray S_ang, Z_ang
    cdef float s

    S_ang, Z_ang = calc_SZ_single(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)

    s = S_ang[i, j].real if real == 1 else S_ang[i, j].imag

    return s * or_pdf(betaDeg)

def ifunc_Z(float betaDeg, float alphaDeg, int i, int j, int real, float nmax, float wavelength, float izaDeg,
            float vzaDeg, float iaaDeg, float vaaDeg, or_pdf):
    cdef np.ndarray S_ang, Z_ang

    S_and, Z_ang = calc_SZ_single(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)
    return Z_ang[i, j] * or_pdf(betaDeg)

# ---- Calculation of namx, S and Z ----
# Calc nmax
cdef float ddelt = 1e-3
cdef int ndgs = 2

cdef calc_nmax(float radius, int radius_type, float wavelength, double complex eps, float axis_ratio, int shape):
    """Initialize the T-matrix.
    """
    return tmatrix.calctmat(radius, radius_type, wavelength, eps.real, eps.imag, axis_ratio, shape, ddelt, ndgs)

# Calc S and Z for different types of orientations
cdef calc_SZ_single(float nmax, float wavelength, float izaDeg, float vzaDeg, float iaaDeg, float vaaDeg,
                    float alphaDeg, float betaDeg):
    return tmatrix.calcampl(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)

cdef orient_averaged_adaptive(float nmax, float wavelength, float izaDeg, float vzaDeg, float iaaDeg, float vaaDeg,
                              or_pdf):
    """Compute the T-matrix using variable orientation scatterers.

    This method uses a very slow adaptive routine and should mainly be used
    for reference purposes. Uses the set particle orientation PDF, ignoring
    the alpha and beta attributes.

    Args:
        : selfatrix (or descendant) instance

    Returns:
        The amplitude (S) and phase (Z) matrices.
    """
    cdef np.ndarray S = np.zeros((2, 2), dtype=complex)
    cdef np.ndarray Z = np.zeros((4, 4))
    cdef int i, j

    ind = range(2)
    for i in ind:
        for j in ind:
            S.real[i, j] = dblquad(ifunc_S, 0.0, 360.0,
                                   lambda x: 0.0, lambda x: 180.0,
                                   (i, j, 1, nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, or_pdf))[0] / 360.0
            S.imag[i, j] = dblquad(ifunc_S, 0.0, 360.0,
                                   lambda x: 0.0, lambda x: 180.0,
                                   (i, j, 0, nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, or_pdf))[0] / 360.0

    ind = range(4)
    for i in ind:
        for j in ind:
            Z[i, j] = dblquad(ifunc_Z, 0.0, 360.0,
                              lambda x: 0.0, lambda x: 180.0,
                              (i, j, 1, nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, or_pdf))[0] / 360.0

    return S, Z

cdef orient_averaged_fixed(float nmax, float wavelength, float izaDeg, float vzaDeg, float iaaDeg, float vaaDeg,
                           int n_alpha, int n_beta, or_pdf):
    """Compute the T-matrix using variable orientation scatterers.

    This method uses a fast Gaussian quadrature and is suitable
    for most use. Uses the set particle orientation PDF, ignoring
    the alpha and beta attributes.

    Args:
        : selfatrix (or descendant) instance.

    Returns:
        The amplitude (S) and phase (Z) matrices.
    """
    cdef np.ndarray S_ang, Z_ang, beta_p, beta_w
    cdef float alpha

    cdef np.ndarray S = np.zeros((2, 2), dtype=complex)
    cdef np.ndarray Z = np.zeros((4, 4))

    cdef np.ndarray ap = np.linspace(0, 360, n_alpha + 1)[:-1]
    cdef float aw = 1.0 / n_alpha

    beta_p, beta_w = get_points_and_weights(or_pdf, 0, 180, num_points=n_beta)

    for alpha in ap:
        for (beta, w) in zip(beta_p, beta_w):
            S_ang, Z_ang = calc_SZ_single(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alpha, beta)
            S += w * S_ang
            Z += w * Z_ang

    cdef float sw = beta_w.sum()
    # normalize to get a proper average
    S *= aw / sw
    Z *= aw / sw

    return S, Z

# ----------------------------------------------------------------------------------------------------------------------
# Scattering and Extinction Cross Section
# ----------------------------------------------------------------------------------------------------------------------
# Scattering intensity
cdef sca_intensity(np.ndarray Z, int pol):
    """Scattering intensity (phase function) for the current setup.

    Args:
        scatterer: a Scatterer instance.
        h_pol: If True (default), use horizontal polarization.
        If False, use vertical polarization.

    Returns:
        The differential scattering cross section.
    """
    cdef float VV, HH
    VV = Z[0, 0] + Z[0, 1]
    HH = Z[0, 0] - Z[0, 1]

    if pol == 1:
        return VV
    if pol == 2:
        return HH

# ----------------------------------------------------------------------------------------------------------------------
# Other Auxiliary Functions
# ----------------------------------------------------------------------------------------------------------------------
cdef equal_volume_from_maximum(float radius, float axis_ratio, int shape):
    cdef float r_eq

    if shape == -1:
        if axis_ratio > 1.0:  # oblate
            r_eq = radius / axis_ratio ** (1.0 / 3.0)
        else:  # prolate
            r_eq = radius / axis_ratio ** (2.0 / 3.0)
    elif shape == -2:
        if axis_ratio > 1.0:  # oblate
            r_eq = radius * (0.75 / axis_ratio) ** (1.0 / 3.0)
        else:  # prolate
            r_eq = radius * (0.75 / axis_ratio) ** (2.0 / 3.0)
    else:
        return -99999

    return r_eq

# ----------------------------------------------------------------------------------------------------------------------
# Scattering Function Call Wrapper
# ----------------------------------------------------------------------------------------------------------------------
def sca_intensity_wrapper(np.ndarray Z, int pol):
    return sca_intensity(Z, pol)

# ----------------------------------------------------------------------------------------------------------------------
# Auxiliary Function Call Wrapper
# ----------------------------------------------------------------------------------------------------------------------
def equal_volume_from_maximum_wrapper(float radius, float axis_ratio, int shape):
    return equal_volume_from_maximum(radius, axis_ratio, shape)

# ----------------------------------------------------------------------------------------------------------------------
# SZ Orientation Call Wrapper
# ----------------------------------------------------------------------------------------------------------------------
def calc_nmax_wrapper(float radius, int radius_type, float wavelength, double complex eps, float axis_ratio, int shape):
    test = calc_nmax(radius, radius_type, wavelength, eps, axis_ratio, shape)

    return test

def calc_single_wrapper(float nmax, float wavelength, float izaDeg, float vzaDeg, float iaaDeg, float vaaDeg,
                        float alphaDeg, float betaDeg):
    return calc_SZ_single(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)

def orient_averaged_adaptive_wrapper(float nmax, float wavelength, float izaDeg, float vzaDeg, float iaaDeg,
                                     float vaaDeg, or_pdf):
    return orient_averaged_adaptive(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, or_pdf)

def orient_averaged_fixed_wrapper(float nmax, float wavelength, float izaDeg, float vzaDeg, float iaaDeg, float vaaDeg,
                                  int n_alpha, int n_beta, or_pdf):
    return orient_averaged_fixed(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, n_alpha, n_beta, or_pdf)
