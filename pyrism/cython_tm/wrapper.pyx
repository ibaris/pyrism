# -*- coding: utf-8 -*-
"""
Created on 28.02.2019 by Ismail Baris
"""
from __future__ import division
from pyrism.cython_tm.core cimport NMAX_VEC
from pyrism.cython_tm.sz_matrix cimport SZ_S_VEC, SZ_AF_VEC
from pyrism.cython_tm.pdf cimport gaussian as c_gaussian
from pyrism.cython_tm.pdf cimport uniform as c_uniform
from pyrism.cython_tm.auxil cimport equal_volume_from_maximum
from pyrism.cython_tm.xsec cimport XE as cXE
from pyrism.cython_tm.xsec cimport XS_S as cXS_S
from pyrism.cython_tm.xsec cimport XS_AF as cXS_AF
from pyrism.cython_tm.xsec cimport XSEC_QSI
# ------------------------------------------------------------------------------------------------------------
# Core Wrapper
# ------------------------------------------------------------------------------------------------------------
def NMAX(double[:] radius, int radius_type, double[:] wavelength, double[:] eps_real, double[:] eps_imag,
         double[:] axis_ratio, int shape, int verbose):
    """
    Calculate NMAX in a vectorized function.

    Parameters
    ----------
    radius : double[:]
        Equivalent particle radius in same unit as wavelength.
    radius_type : int, {0, 1, 2}
        Specification of particle radius:
            * 0: radius is the equivalent volume radius (default).
            * 1: radius is the equivalent area radius.
            * 2: radius is the maximum radius.
    wavelength : double[:]
        Wavelength in same unit as radius.
    eps_real, eps_imag : double[:]
        Real and imaginary part of the dielectric constant.
    axis_ratio : double[:]
        The horizontal-to-rotational axis ratio.
    shape : int, {-1, -2}
        Shape of the particle:
            * -1 : spheroid,
            * -2 : cylinders.

    Returns
    -------
    nmax : MemoryView

    """

    return NMAX_VEC(radius=radius, radius_type=radius_type, wavelength=wavelength, eps_real=eps_real, eps_imag=eps_imag,
                axis_ratio=axis_ratio, shape=shape, verbose=verbose)

# ------------------------------------------------------------------------------------------------------------
# Wrapper for S and Z
# ------------------------------------------------------------------------------------------------------------
# S and Z for Single Orientation ---------------------------------------------------------------------------------------
def SZ_S(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] vzaDeg, double[:] iaaDeg,
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
    return SZ_S_VEC(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)

# S and Z for Adaptive Fixed Orientation -------------------------------------------------------------------------------
def SZ_AF(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] vzaDeg, double[:] iaaDeg,
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

    return SZ_AF_VEC(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, n_alpha, n_beta, or_pdf)

# ----------------------------------------------------------------------------------------------------------------------
# Orientation Wrapper
# ----------------------------------------------------------------------------------------------------------------------
def gaussian(float std, float mean):
    """Gaussian probability distribution function (PDF) for orientation averaging.

    Parameters
    ----------
    std : float
        The standard deviation in degrees of the Gaussian PDF
    mean : float
        The mean in degrees of the Gaussian PDF.  This should be a number in the interval [0, 180)

    Returns
    -------
    pdf(x): callable
        A function that returns the value of the spherical Jacobian-normalized Gaussian PDF with the given STD at x
        (degrees). It is normalized for the interval [0, 180].
    """
    return c_gaussian(std, mean)

def uniform():
    """Uniform probability distribution function (PDF) for orientation averaging.

    Returns
    -------
    pdf(x): callable
        A function that returns the value of the spherical Jacobian-normalized uniform PDF. It is normalized for
        the interval [0, 180].
    """
    return c_uniform()

# ----------------------------------------------------------------------------------------------------------------------
# Auxiliary Wrapper
# ----------------------------------------------------------------------------------------------------------------------
def EVFM(double[:] radius, double[:] axis_ratio, int shape):
    """
    Convert maximum radius into volume equivalent radius

    Parameters
    ----------
    radius : double[:]
        Equivalent particle radius in same unit as wavelength.
    axis_ratio : double[:]
        The horizontal-to-rotational axis ratio.
    shape : int, {-1, -2}
        Shape of the particle:
            * -1 : spheroid,
            * -2 : cylinders.

    Returns
    -------
    Rv : MemoryView, double[:]
        Volume equivalent radius.
    """
    return equal_volume_from_maximum(radius, axis_ratio, shape)

# ----------------------------------------------------------------------------------------------------------------------
# Cross Section Wrapper
# ----------------------------------------------------------------------------------------------------------------------
# Single Orientation ---------------------------------------------------------------------------------------------------
# Extinction and Intensity ---------------------------------------------------------------------------------------------
def XE(double complex[:,:,:] S, double[:] wavelength):
    """Extinction Cross Section.

    Parameters
    ----------
    S : double complex[:,:,:]
        Three dimensoinal scattering matrix.
    wavelength : double[:]
        Wavelength in same unit as radius (used by function calc_nmax).

    Returns
    -------
    Ke : double[:,:,:]
        Three dimensional extinction matrix.
    """
    return cXE(S, wavelength)

def XS_S(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg,
       double[:] alphaDeg, double[:] betaDeg, int Nx, int Ny, int verbose):
    """Scattering Cross Section MATRIX for single orientation.

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
    N : int, double[:]
        Number of scatterer in unit volume.
    Nx, Ny : int
        Integration steps.
    verbose : int
        If 1 verbose mode is on.

    Returns
    -------
    Qs : double[:,:,:]
        Two dimensional scattering cross section.
    """

    return cXS_S(nmax, wavelength, izaDeg, iaaDeg, alphaDeg, betaDeg, Nx, Ny, verbose, 0)

def XI(double[:,:,:] Z):
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
    return XSEC_QSI(Z)

# Averaged Fixed Orientation -------------------------------------------------------------------------------------------
def XS_AF(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg, int Nx, int Ny, int n_alpha,
          int n_beta, or_pdf, int verbose):
    """Scattering Cross Section for Averaged Fixed Orientation.

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
    Qs : double[:,:]
        Two dimensional scattering cross section.
    """

    return cXS_AF(nmax, wavelength, izaDeg, iaaDeg, Nx, Ny, n_alpha, n_beta, or_pdf, verbose, 0)

def XASY_S(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg,
       double[:] alphaDeg, double[:] betaDeg, int Nx, int Ny, int verbose):
    """Asymmetry Cross Section MATRIX for single orientation.

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
    N : int, double[:]
        Number of scatterer in unit volume.
    Nx, Ny : int
        Integration steps.
    verbose : int
        If 1 verbose mode is on.

    Returns
    -------
    Qs : double[:,:,:]
        Two dimensional scattering cross section.
    """

    return cXS_S(nmax, wavelength, izaDeg, iaaDeg, alphaDeg, betaDeg, Nx, Ny, verbose, 1)



# Averaged Fixed Orientation -------------------------------------------------------------------------------------------
def XASY_AF(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg, int Nx, int Ny, int n_alpha, int n_beta,
          or_pdf, int verbose):
    """Asymmetry Cross Section for Averaged Fixed Orientation.

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
    Qs : double[:,:]
        Two dimensional scattering cross section.
    """

    return cXS_AF(nmax, wavelength, izaDeg, iaaDeg, Nx, Ny, n_alpha, n_beta, or_pdf, verbose, 1)
