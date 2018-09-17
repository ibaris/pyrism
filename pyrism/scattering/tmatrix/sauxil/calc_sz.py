from __future__ import division

import numpy as np
from scipy.integrate import dblquad

from pyrism.fortran_tm import fotm as tmatrix
from .quadrature import get_points_and_weights


def equal_volume_from_maximum(radius, axis_ratio, shape):
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
        raise AttributeError("Unsupported shape for maximum radius.")
    return r_eq


def calc_nmax(radius, radius_type, wavelength, eps, axis_ratio, shape):
    """Initialize the T-matrix.
    """
    m = eps
    ddelt = 1e-3
    ndgs = 2

    if radius_type == 2.0:
        # Maximum radius is not directly supported in the original
        # so we convert it to equal volume radius
        radius_type = 1.0
        radius = equal_volume_from_maximum()
    else:
        radius_type = radius_type
        radius = radius

    return tmatrix.calctmat(radius, radius_type, wavelength, m.real, m.imag, axis_ratio, shape, ddelt, ndgs)


def calc_SZ_single(nmax, wavelength, iza, vza, iaa, vaa, alpha, beta):
    return tmatrix.calcampl(nmax, wavelength, iza, vza, iaa, vaa, alpha, beta)


def orient_averaged_adaptive(nmax, wavelength, iza, vza, iaa, vaa, or_pdf):
    """Compute the T-matrix using variable orientation scatterers.

    This method uses a very slow adaptive routine and should mainly be used
    for reference purposes. Uses the set particle orientation PDF, ignoring
    the alpha and beta attributes.

    Args:
        : selfatrix (or descendant) instance

    Returns:
        The amplitude (S) and phase (Z) matrices.
    """
    S = np.zeros((2, 2), dtype=complex)
    Z = np.zeros((4, 4))

    def Sfunc(beta, alpha, i, j, real):
        (S_ang, Z_ang) = calc_SZ_single(nmax, wavelength, iza, vza, iaa, vaa, alpha, beta)
        s = S_ang[i, j].real if real else S_ang[i, j].imag
        return s * or_pdf(beta)

    ind = range(2)
    for i in ind:
        for j in ind:
            S.real[i, j] = dblquad(Sfunc, 0.0, 360.0,
                                   lambda x: 0.0, lambda x: 180.0, (i, j, True))[0] / 360.0
            S.imag[i, j] = dblquad(Sfunc, 0.0, 360.0,
                                   lambda x: 0.0, lambda x: 180.0, (i, j, False))[0] / 360.0

    def Zfunc(beta, alpha, i, j):
        (S_and, Z_ang) = calc_SZ_single(nmax, wavelength, iza, vza, iaa, vaa, alpha, beta)
        return Z_ang[i, j] * or_pdf(beta)

    ind = range(4)
    for i in ind:
        for j in ind:
            Z[i, j] = dblquad(Zfunc, 0.0, 360.0,
                              lambda x: 0.0, lambda x: 180.0, (i, j))[0] / 360.0

    return (S, Z)


def orient_averaged_fixed(nmax, wavelength, iza, vza, iaa, vaa, n_alpha, n_beta, or_pdf):
    """Compute the T-matrix using variable orientation scatterers.

    This method uses a fast Gaussian quadrature and is suitable
    for most use. Uses the set particle orientation PDF, ignoring
    the alpha and beta attributes.

    Args:
        : selfatrix (or descendant) instance.

    Returns:
        The amplitude (S) and phase (Z) matrices.
    """
    S = np.zeros((2, 2), dtype=complex)
    Z = np.zeros((4, 4))
    ap = np.linspace(0, 360, n_alpha + 1)[:-1]
    aw = 1.0 / n_alpha

    beta_p, beta_w = get_points_and_weights(or_pdf, 0, 180, n_beta)

    for alpha in ap:
        for (beta, w) in zip(beta_p, beta_w):
            (S_ang, Z_ang) = calc_SZ_single(nmax, wavelength, iza, vza, iaa, vaa, alpha, beta)
            S += w * S_ang
            Z += w * Z_ang

    sw = beta_w.sum()
    # normalize to get a proper average
    S *= aw / sw
    Z *= aw / sw

    return (S, Z)


def calc_SZ_orient(nmax, wavelength, iza, vza, iaa, vaa, alpha, beta, n_alpha, n_beta, or_pdf, orientation):
    if orientation is 'S':
        return calc_SZ_single(nmax, wavelength, iza, vza, iaa, vaa, alpha, beta)
    elif orientation is 'AA':
        return orient_averaged_adaptive(nmax, wavelength, iza, vza, iaa, vaa, or_pdf)

    elif orientation is 'AF':
        return orient_averaged_fixed(nmax, wavelength, iza, vza, iaa, vaa, n_alpha, n_beta, or_pdf)
    else:
        raise AttributeError("The parameter orient must be 'S', 'AA' or 'AF'")
