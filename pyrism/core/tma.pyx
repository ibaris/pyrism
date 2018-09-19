# cython: cdivision=True
from __future__ import division

cimport numpy as np
import numpy as np
from pyrism.auxil.quadrature import get_points_and_weights

DTYPE = np.float
from libc.math cimport sin, cos, pi

ctypedef np.float_t DTYPE_t
from scipy.integrate import dblquad
from pyrism.fortran_tm import fotm as tmatrix

deg_to_rad = pi / 180.0

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

cdef calc_nmax(float radius, int radius_type, float wavelength, double complex eps, float axis_ratio, int shape):
    """Initialize the T-matrix.
    """

    cdef float ddelt = 1e-3
    cdef int ndgs = 2

    if radius_type == 2:
        # Maximum radius is not directly supported in the original
        # so we convert it to equal volume radius
        radius_type = 1
        radius = equal_volume_from_maximum(radius, axis_ratio, shape)
    else:
        radius_type = radius_type
        radius = radius

    return tmatrix.calctmat(radius, radius_type, wavelength, eps.real, eps.imag, axis_ratio, shape, ddelt, ndgs)

cdef calc_SZ_single(float nmax, float wavelength, float iza, float vza, float iaa, float vaa, float alpha, float beta):
    return tmatrix.calcampl(nmax, wavelength, iza, vza, iaa, vaa, alpha, beta)

cdef Sfunc(float beta, float alpha, int i, int j, int real, float nmax, float wavelength, float iza, float vza,
           float iaa, float vaa, or_pdf):
    cdef np.ndarray S_ang, Z_ang
    cdef float s

    S_ang, Z_ang = calc_SZ_single(nmax, wavelength, iza, vza, iaa, vaa, alpha, beta)

    s = S_ang[i, j].real if real == 1 else S_ang[i, j].imag

    return s * or_pdf(beta)

def Zfunc(float beta, float alpha, int i, int j, int real, float nmax, float wavelength, float iza, float vza,
          float iaa, float vaa, or_pdf):
    cdef np.ndarray S_ang, Z_ang

    S_and, Z_ang = calc_SZ_single(nmax, wavelength, iza, vza, iaa, vaa, alpha, beta)
    return Z_ang[i, j] * or_pdf(beta)

cdef orient_averaged_adaptive(float nmax, float wavelength, float iza, float vza, float iaa, float vaa, or_pdf):
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
            S.real[i, j] = dblquad(Sfunc, 0.0, 360.0,
                                   lambda x: 0.0, lambda x: 180.0,
                                   (i, j, 1, nmax, wavelength, iza, vza, iaa, vaa, or_pdf))[0] / 360.0
            S.imag[i, j] = dblquad(Sfunc, 0.0, 360.0,
                                   lambda x: 0.0, lambda x: 180.0,
                                   (i, j, 0, nmax, wavelength, iza, vza, iaa, vaa, or_pdf))[0] / 360.0

    ind = range(4)
    for i in ind:
        for j in ind:
            Z[i, j] = dblquad(Zfunc, 0.0, 360.0,
                              lambda x: 0.0, lambda x: 180.0,
                              (i, j, 1, nmax, wavelength, iza, vza, iaa, vaa, or_pdf))[0] / 360.0

    return S, Z

cdef orient_averaged_fixed(float nmax, float wavelength, float iza, float vza, float iaa, float vaa, int n_alpha,
                           int n_beta, or_pdf):
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
            S_ang, Z_ang = calc_SZ_single(nmax, wavelength, iza, vza, iaa, vaa, alpha, beta)
            S += w * S_ang
            Z += w * Z_ang

    cdef float sw = beta_w.sum()
    # normalize to get a proper average
    S *= aw / sw
    Z *= aw / sw

    return S, Z

cdef iufun_sca_xsect(float vza, float vaa, float nmax, float wavelength, float izaDeg, float iaaDeg, float alphaDeg,
                     float betaDeg, int n_alpha, int n_beta, or_pdf, orient, int pol):
    """Get the S and Z matrices for a single orientation.
    """

    cdef np.ndarray S, Z

    vza, vaa = vza * (180.0 / np.pi), vaa * (180.0 / np.pi)

    S, Z = get_oriented_SZ(nmax, wavelength, izaDeg, vza, iaaDeg, vaa, alphaDeg, betaDeg, n_alpha, n_beta, or_pdf,
                           orient)

    if pol == 1:
        return (Z[0, 0] + Z[0, 1]) * sin((np.pi / 180.0) * vza)
    if pol == 2:
        return (Z[0, 0] - Z[0, 1]) * sin((np.pi / 180.0) * vza)

cdef integrate_ifun_sca_xsect(float nmax, float wavelength, float izaDeg, float iaaDeg,
                              float alphaDeg, float betaDeg, int n_alpha, int n_beta, or_pdf, orient):
    cdef float xsectVV, xsectHH

    xsectVV = dblquad(iufun_sca_xsect, 0.0, 2 * np.pi, lambda x: 0.0, lambda x: np.pi,
                      args=(nmax, wavelength, izaDeg, iaaDeg, alphaDeg,
                            betaDeg, n_alpha, n_beta, or_pdf, orient, 1))[0]

    xsectHH = dblquad(iufun_sca_xsect, 0.0, 2 * np.pi, lambda x: 0.0, lambda x: np.pi,
                      args=(nmax, wavelength, izaDeg, iaaDeg, alphaDeg,
                            betaDeg, n_alpha, n_beta, or_pdf, orient, 2))[0]

    return xsectVV, xsectHH

cdef iufun_asy(float vza, float vaa, float cos_t0, float sin_t0, float nmax, float wavelength, float izaDeg,
               float iaaDeg, float alphaDeg,
               float betaDeg, int n_alpha, int n_beta, or_pdf, orient, int pol):
    """Get the S and Z matrices for a single orientation.
    """

    cdef np.ndarray S, Z
    cdef float cos_T_sin_t, vzaDeg, iaa

    vzaDeg, vaaDeg = vza * (180.0 / np.pi), vaa * (180.0 / np.pi)
    iaa = iaaDeg * deg_to_rad

    S, Z = get_oriented_SZ(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg, n_alpha, n_beta, or_pdf,
                           orient)

    cos_T_sin_t = 0.5 * (np.sin(2 * vza) * cos_t0 + (1 - np.cos(2 * vza)) * sin_t0 * np.cos(iaa - vaa))

    if pol == 1:
        return (Z[0, 0] + Z[0, 1]) * cos_T_sin_t
    if pol == 2:
        return (Z[0, 0] - Z[0, 1]) * cos_T_sin_t

cdef asym(float nmax, float wavelength, float izaDeg, float iaaDeg, float alphaDeg,
          float betaDeg, int n_alpha, int n_beta, or_pdf, orient):
    """Asymmetry parameter for the current setup, with polarization.

    Args:
        scatterer: a Scatterer instance.
        h_pol: If True (default), use horizontal polarization.
        If False, use vertical polarization.

    Returns:
        The asymmetry parameter.
    """
    cdef float cos_t0, sin_t0, p0, cos_int_VV, cos_int_HH, sca_xsecVV, sca_xsecHH

    cos_t0 = np.cos(izaDeg * deg_to_rad)
    sin_t0 = np.sin(izaDeg * deg_to_rad)
    p0 = iaaDeg * deg_to_rad

    cos_int_VV = dblquad(iufun_asy, 0.0, 2 * np.pi, lambda x: 0.0, lambda x: np.pi,
                         args=(cos_t0, sin_t0, nmax, wavelength, izaDeg, iaaDeg, alphaDeg,
                               betaDeg, n_alpha, n_beta, or_pdf, orient, 1))[0]

    cos_int_HH = dblquad(iufun_asy, 0.0, 2 * np.pi, lambda x: 0.0, lambda x: np.pi,
                         args=(cos_t0, sin_t0, nmax, wavelength, izaDeg, iaaDeg, alphaDeg,
                               betaDeg, n_alpha, n_beta, or_pdf, orient, 2))[0]

    sca_xsecVV, sca_xsecHH = integrate_ifun_sca_xsect(nmax, wavelength, izaDeg, iaaDeg, alphaDeg, betaDeg, n_alpha,
                                                      n_beta, or_pdf,
                                                      orient)

    return cos_int_VV / sca_xsecVV, cos_int_HH / sca_xsecHH

def ext_xsect(float nmax, float wavelength, float iza, float vza, float iaa, float vaa, float alpha, float beta,
              int n_alpha, int n_beta, or_pdf, orientation):
    cdef np.ndarray S, Z
    cdef float VV, HH

    vza = iza
    vaa = iaa

    S, Z = get_oriented_SZ(nmax, wavelength, iza, vza, iaa, vaa, alpha, beta, n_alpha, n_beta, or_pdf,
                           orientation)

    VV = 2 * wavelength * S[0, 0].imag
    HH = 2 * wavelength * S[1, 1].imag

    return VV, HH

def asm_wrapper(float nmax, float wavelength, float izaDeg, float iaaDeg, float alphaDeg,
                float betaDeg, int n_alpha, int n_beta, or_pdf, orient):
    return asym(nmax, wavelength, izaDeg, iaaDeg, alphaDeg, betaDeg, n_alpha, n_beta, or_pdf, orient)

def sca_xsect_wrapper(float nmax, float wavelength, float izaDeg, float iaaDeg, float alphaDeg,
                      float betaDeg, int n_alpha, int n_beta, or_pdf, orient):
    return integrate_ifun_sca_xsect(nmax, wavelength, izaDeg, iaaDeg, alphaDeg,
                                    betaDeg, n_alpha, n_beta, or_pdf, orient)

def get_oriented_SZ(float nmax, float wavelength, float iza, float vza, float iaa, float vaa, float alpha, float beta,
                    int n_alpha, int n_beta, or_pdf, orientation):
    if orientation is 'S':
        return calc_SZ_single(nmax, wavelength, iza, vza, iaa, vaa, alpha, beta)
    elif orientation is 'AA':
        return orient_averaged_adaptive(nmax, wavelength, iza, vza, iaa, vaa, or_pdf)

    elif orientation is 'AF':
        return orient_averaged_fixed(nmax, wavelength, iza, vza, iaa, vaa, n_alpha, n_beta, or_pdf)
    else:
        return -99999

def calc_nmax_wrapper(float radius, int radius_type, float wavelength, double complex eps, float axis_ratio, int shape):
    test = calc_nmax(radius, radius_type, wavelength, eps, axis_ratio, shape)

    return test
