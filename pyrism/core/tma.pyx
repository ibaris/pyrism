# cython: cdivision=True
from __future__ import division

cimport numpy as np
import numpy as np
from pyrism.auxil.quadrature import get_points_and_weights

from libc.math cimport sin

from scipy.integrate import dblquad, quad
from pyrism.fortran_tm.fotm import calctmat, calcampl

DTYPE = np.complex
ctypedef np.complex_t DTYPE_t

# ----------------------------------------------------------------------------------------------------------------------
# Todo and Comments
# ----------------------------------------------------------------------------------------------------------------------
# TODO 21.10.18 ibaris: This is anoying. If I try to calculate nmax out of the loop, the unit tests fails.
# TODO 26.10.18 ibaris: In function IFUNC_SZ_S_VZA, IFUNC_XSEC_QS_S the definition of betaDeg caused a compiler error?
# TODO 26.10.18 ibaris: In function IFUNC_SZ_AF_VZA, IFUNC_SZ_AF_IZA the definition of n_alpha caused a compiler error?
# The PSD integrator is not programmed yet. Moreover, the AA orientation will be deleted.

# ----------------------------------------------------------------------------------------------------------------------
# Set Auxiliary Variables
# ----------------------------------------------------------------------------------------------------------------------
cdef :
    float ddelt = 1e-3
    int ndgs = 2
    float PI = 3.14159265359
    float DEG_TO_RAD = PI / 180.0
    float RAD_TO_DEG = 180.0 / PI

# ----------------------------------------------------------------------------------------------------------------------
# NMAX
# ----------------------------------------------------------------------------------------------------------------------
# Vectorized -----------------------------------------------------------------------------------------------------------
cdef int[:] NMAX_VEC(double[:] radius, int radius_type, double[:] wavelength, double[:] eps_real, double[:] eps_imag,
                     double[:] axis_ratio, int shape):
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

    cdef:
        Py_ssize_t xmax = radius.shape[0]
        Py_ssize_t x

    result = np.zeros_like(radius, dtype=np.intc)
    cdef int[:] result_view = result

    for x in range(xmax):
        result_view[x] = calctmat(radius[x], radius_type, wavelength[x], eps_real[x], eps_imag[x], axis_ratio[x],
                                  shape, ddelt, ndgs)

    return result

# NOT Vectorized -------------------------------------------------------------------------------------------------------
cdef int NMAX(double radius, int radius_type, double wavelength, double eps_real, double eps_imag, double axis_ratio,
              int shape):
    """
    Calculate NMAX.
    
    Parameters
    ----------
    radius : double
        Equivalent particle radius in same unit as wavelength.
    radius_type : int, {0, 1, 2}
        Specification of particle radius:
            * 0: radius is the equivalent volume radius (default).
            * 1: radius is the equivalent area radius.
            * 2: radius is the maximum radius.
    wavelength : double
        Wavelength in same unit as radius.
    eps_real, eps_imag : double
        Real and imaginary part of the dielectric constant.
    axis_ratio : double
        The horizontal-to-rotational axis ratio.
    shape : int, {-1, -2}
        Shape of the particle:
            * -1 : spheroid,
            * -2 : cylinders.
    
    Returns
    -------
    nmax : int
    
    """
    return calctmat(radius, radius_type, wavelength, eps_real, eps_imag, axis_ratio, shape, ddelt, ndgs)

# ----------------------------------------------------------------------------------------------------------------------
# T-Matrix to Calculate S, Z
# ----------------------------------------------------------------------------------------------------------------------
# Single Orientation ---------------------------------------------------------------------------------------------------
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

# Adaptive Average Fixed Orientation -----------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
# Integrate SINGLE ORIENTATED Scattering and Phase Matrix
# ----------------------------------------------------------------------------------------------------------------------
# Auxiliary Functions --------------------------------------------------------------------------------------------------
# Functions that can be used with scipy.integrate.dblquad
cdef double IFUNC_SZ_S_IZA(double iza, double iaa, int nmax, double wavelength, double vzaDeg, double vaaDeg,
                        double alphaDeg, double betaDeg, int i):
    """Integrate Z in Incidence Direction
    
    Callable function for scipy.integrate.dblquad for integration of the phase matrix in incidence direction.

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
    i : int
        Number between 0 and 15 which defines the matrix element that will be integrated.
        
    Returns
    -------
    Z : double
        Integrated phase matrix element.
    """

    cdef:
        double izaDeg, iaaDeg
        np.ndarray[double, ndim=2]  Z
        np.ndarray[DTYPE_t, ndim=2]  S

    izaDeg, iaaDeg = iza * RAD_TO_DEG, iaa * RAD_TO_DEG

    S, Z = SZ_S(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)

    return Z.flatten()[i] * sin(iza)

cdef double IFUNC_SZ_S_VZA(double vza, double vaa, int nmax, double wavelength, double izaDeg, double iaaDeg,
                           double alphaDeg, betaDeg, int i):
    """Integrate Z in Viewing Direction
    
    Callable function for scipy.integrate.dblquad for integration of the phase matrix in viewing direction.

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
    i : int
        Number between 0 and 15 which defines the matrix element that will be integrated.
        
    Returns
    -------
    Z : double
        Integrated phase matrix element.
    """
    cdef:
        double vzaDeg, vaaDeg
        np.ndarray[double, ndim=2]  Z
        np.ndarray[DTYPE_t, ndim=2]  S

    vzaDeg, vaaDeg = vza * RAD_TO_DEG, vaa * RAD_TO_DEG

    S, Z = SZ_S(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)

    return Z.flatten()[i] * sin(vza)

# Double Integral Functions --------------------------------------------------------------------------------------------
cdef double[:,:] DBLQUAD_Z_S_IZA(int[:] nmax, double[:] wavelength, double[:] vzaDeg, double[:] vaaDeg, double[:] alphaDeg,
                                 double[:] betaDeg):
    """
    Integrate the phase matrix over a half sphere in incidence direction.
    
    Parameters
    ----------
    nmax : int[:] 
        Nmax parameter.
    wavelength : double[:]
        Wavelength in same unit as radius (used by function calc_nmax).
    vzaDeg, vaaDeg : double[:]
        Scattering (vza) zenith and azimuth angle (vaa) in [DEG].
    alphaDeg, betaDeg: double[:]
        The Euler angles of the particle orientation in [DEG].
        
    Returns
    -------
    Z : MemoryView, double[:,:]
        Two dimensional integrated phase matrix.
    
    Note
    ----
    The output is a array with shape (nmax.shape[0], 16). To put it in a matrix form one must reshape it like
    np.reshape(Z, (nmax.shape[0], 4, 4)).
    """

    cdef:
        Py_ssize_t x, i
        Py_ssize_t xmax = nmax.shape[0]

    Z = np.zeros((xmax, 16), dtype=np.double)
    cdef double[:,:] Z_view = Z

    for x in range(xmax):
            for i in range(16):
                Z_view[x, i] = dblquad(IFUNC_SZ_S_IZA, 0, 2 * PI, lambda x: 0, lambda x: PI,
                                       args=(nmax[x], wavelength[x], vzaDeg[x], vaaDeg[x], alphaDeg[x], betaDeg[x],
                                             i))[0]


    return Z

cdef double[:,:] DBLQUAD_Z_S_VZA(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg, double[:] alphaDeg,
                                 double[:] betaDeg):
    """
    Integrate the phase matrix over a half sphere in vieving direction.
    
    Parameters
    ----------
    nmax : int[:] 
        Nmax parameter.
    wavelength : double[:]
        Wavelength in same unit as radius (used by function calc_nmax).
    izaDeg, iaaDeg : double[:]
        Incidence (iza) zenith and azimuth angle (iaa) in [DEG].
    alphaDeg, betaDeg: double[:]
        The Euler angles of the particle orientation in [DEG].
        
    Returns
    -------
    Z : MemoryView, double[:,:]
        Two dimensional integrated phase matrix.
    
    Note
    ----
    The output is a array with shape (nmax.shape[0], 16). To put it in a matrix form one must reshape it like
    np.reshape(Z, (nmax.shape[0], 4, 4)).
    """

    cdef:
        Py_ssize_t x, i
        Py_ssize_t xmax = nmax.shape[0]

    Z = np.zeros((xmax, 16), dtype=np.double)
    cdef double[:,:] Z_view = Z

    for x in range(xmax):
        for i in range(16):
            Z_view[x, i] = dblquad(IFUNC_SZ_S_VZA, 0, 2 * PI, lambda x: 0, lambda x: PI,
                                   args=(nmax[x], wavelength[x], izaDeg[x], iaaDeg[x], alphaDeg[x], betaDeg[x],
                                         i))[0]

    return Z

# ----------------------------------------------------------------------------------------------------------------------
# Integrate AVERAGED FIXED ORIENTATED Scattering and Phase Matrix
# ----------------------------------------------------------------------------------------------------------------------
# Auxiliary Functions --------------------------------------------------------------------------------------------------
# Functions that can be used with scipy.integrate.dblquad
cdef double IFUNC_SZ_AF_IZA(double iza, double iaa, int nmax, double wavelength, double vzaDeg, double vaaDeg,
                            int n_alpha, int n_beta, or_pdf, int i):
    """Integrate Z in Incidence Direction
    
    Callable function for scipy.integrate.dblquad for integration of the phase matrix in incidence direction.

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
    i : int
        Number between 0 and 15 which defines the matrix element that will be integrated.
        
    Returns
    -------
    Z : double
        Integrated phase matrix element.
    """
    cdef:
        double izaDeg, iaaDeg
        np.ndarray[double, ndim=2]  Z
        np.ndarray[DTYPE_t, ndim=2]  S

    izaDeg, iaaDeg = iza * RAD_TO_DEG, iaa * RAD_TO_DEG

    S, Z = SZ_AF(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, n_alpha, n_beta, or_pdf)

    return Z.flatten()[i] * sin(iza)

cdef double IFUNC_SZ_AF_VZA(double vza, double vaa, int nmax, double wavelength, double izaDeg, double iaaDeg,
                            n_alpha, int n_beta, or_pdf, int i):
    """Integrate Z in Viewing Direction
    
    Callable function for scipy.integrate.dblquad for integration of the phase matrix in viewing direction.

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
    i : int
        Number between 0 and 15 which defines the matrix element that will be integrated.
        
    Returns
    -------
    Z : double
        Integrated phase matrix element.
    """
    cdef:
        double vzaDeg, vaaDeg
        np.ndarray[double, ndim=2]  Z
        np.ndarray[DTYPE_t, ndim=2]  S

    vzaDeg, vaaDeg = vza * RAD_TO_DEG, vaa * RAD_TO_DEG

    S, Z = SZ_AF(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, n_alpha, n_beta, or_pdf)

    return Z.flatten()[i] * sin(vza)

# Double Integral Functions --------------------------------------------------------------------------------------------
cdef double[:,:] DBLQUAD_Z_AF_IZA(int[:] nmax, double[:] wavelength, double[:] vzaDeg, double[:] vaaDeg,
                                  int n_alpha, int n_beta, or_pdf):
    """
    Integrate the phase matrix over a half sphere in incidence direction.
    
    Parameters
    ----------
    nmax : int[:] 
        Nmax parameter.
    wavelength : double[:]
        Wavelength in same unit as radius (used by function calc_nmax).
    vzaDeg, vaaDeg : double[:]
        Scattering (vza) zenith and azimuth angle (vaa) in [DEG].
    n_alpha, n_beta : int
        Number of integration points in the alpha and beta Euler angle.
    or_pdf: callable
        Particle orientation Probability Density Function (PDF) for orientation averaging.
        
    Returns
    -------
    Z : MemoryView, double[:,:]
        Two dimensional integrated phase matrix.
    
    Note
    ----
    The output is a array with shape (nmax.shape[0], 16). To put it in a matrix form one must reshape it like
    np.reshape(Z, (nmax.shape[0], 4, 4)).
    """

    cdef:
        Py_ssize_t x, i
        Py_ssize_t xmax = nmax.shape[0]

    Z = np.zeros((xmax, 16), dtype=np.double)
    cdef double[:,:] Z_view = Z

    for x in range(xmax):
            for i in range(16):
                Z_view[x, i] = dblquad(IFUNC_SZ_AF_IZA, 0, 2 * PI, lambda x: 0, lambda x: PI,
                                       args=(nmax[x], wavelength[x], vzaDeg[x], vaaDeg[x], n_alpha, n_beta, or_pdf,
                                             i))[0]

    return Z

cdef double[:,:] DBLQUAD_Z_AF_VZA(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg,
                                  int n_alpha, int n_beta, or_pdf):
    """
    Integrate the phase matrix over a half sphere in vieving direction.
    
    Parameters
    ----------
    nmax : int[:] 
        Nmax parameter.
    wavelength : double[:]
        Wavelength in same unit as radius (used by function calc_nmax).
    izaDeg, iaaDeg : double[:]
        Incidence (iza) zenith and azimuth angle (iaa) in [DEG].
    n_alpha, n_beta : int
        Number of integration points in the alpha and beta Euler angle.
    or_pdf: callable
        Particle orientation Probability Density Function (PDF) for orientation averaging.
        
    Returns
    -------
    Z : MemoryView, double[:,:]
        Two dimensional integrated phase matrix.
    
    Note
    ----
    The output is a array with shape (nmax.shape[0], 16). To put it in a matrix form one must reshape it like
    np.reshape(Z, (nmax.shape[0], 4, 4)).
    """

    cdef:
        Py_ssize_t x, i
        Py_ssize_t xmax = nmax.shape[0]

    Z = np.zeros((xmax, 16), dtype=np.double)
    cdef double[:,:] Z_view = Z

    for x in range(xmax):
            for i in range(16):
                Z_view[x, i] = dblquad(IFUNC_SZ_AF_VZA, 0, 2 * PI, lambda x: 0, lambda x: PI,
                                       args=(nmax[x], wavelength[x], izaDeg[x], iaaDeg[x], n_alpha, n_beta, or_pdf,
                                             i))[0]

    return Z

# ----------------------------------------------------------------------------------------------------------------------
# Scattering and Asymetriy Cross Section for SINGLE ORIENTATION
# ----------------------------------------------------------------------------------------------------------------------
# Auxiliary Functions --------------------------------------------------------------------------------------------------
cdef double IFUNC_XSEC_QS_S(double vza, double vaa, int nmax, double wavelength, double izaDeg, double iaaDeg,
                            double alphaDeg, betaDeg, int i, int cos_T_sin_t):
    """Integrate Z in Viewing Direction
    
    Callable function for scipy.integrate.dblquad for integration of the phase matrix in viewing direction.

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
    i : int
        Number between 0 and 15 which defines the matrix element that will be integrated.
    cos_T_sin_t : int
        True for the calculation of the asymetry factor. This adds another term as sin(vza) to the end of the 
        integration.
        
    Returns
    -------
    Z : double
        Integrated phase matrix element.
    """
    cdef:
        double vzaDeg, vaaDeg, cos_t0, sin_t0, multi
        np.ndarray[double, ndim=2]  Z
        np.ndarray[DTYPE_t, ndim=2]  S
        np.ndarray[double, ndim=1]  Zf, I

    vzaDeg, vaaDeg = vza * RAD_TO_DEG, vaa * RAD_TO_DEG

    S, Z = SZ_S(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)

    Zf = Z.flatten()

    I = np.zeros(2, dtype=np.double)

    I[0] = Zf[0] + Zf[1]
    I[1] = Zf[0] - Zf[1]

    if cos_T_sin_t:
        cos_t0 = np.cos(izaDeg * DEG_TO_RAD)
        sin_t0 = np.sin(izaDeg * DEG_TO_RAD)

        multi = 0.5 * (np.sin(2 * vza) * cos_t0 +
                       (1 - np.cos(2 * vza)) * sin_t0 * np.cos((iaaDeg * DEG_TO_RAD) - vaa))

    else:
        multi = sin(vza)

    return I[i] * multi

# Scattering and Asymmetry for Single Orientation ----------------------------------------------------------------------
cdef double[:,:] XSEC_QS_S(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg, double[:] alphaDeg,
                           double[:] betaDeg):
    """Scattering Cross Section for single orientation.
    
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
    Qs : double[:,:]
        Two dimensional scattering cross section.
    """

    cdef:
        int i
        Py_ssize_t x
        Py_ssize_t xmax = nmax.shape[0]

    QS = np.zeros((xmax, 2), dtype=np.double)
    cdef double[:,:] QS_view = QS

    for x in range(xmax):
        for i in range(2):
            QS_view[x, i] = dblquad(IFUNC_XSEC_QS_S, 0, 2 * PI, lambda x: 0, lambda x: PI,
                                           args=(nmax[x], wavelength[x], izaDeg[x], iaaDeg[x], alphaDeg[x], betaDeg[x],
                                                 i, 0))[0]

    return QS_view

cdef double[:,:] XSEC_ASY_S(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg, double[:] alphaDeg,
                            double[:] betaDeg):
    """Asymetry Cross Section for single orientation.
    
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
    Qas : double[:,:]
        Two dimensional asymmetry cross section.
        
    Note
    ----
    To obtain the asymetry factor the output of this function must be divided through the scattering cross section.
    """

    cdef:
        int i
        Py_ssize_t x
        Py_ssize_t xmax = nmax.shape[0]

    QS = np.zeros((xmax, 2), dtype=np.double)
    cdef double[:,:] QS_view = QS

    for x in range(xmax):
        for i in range(2):
            QS_view[x, i] = dblquad(IFUNC_XSEC_QS_S, 0, 2 * PI, lambda x: 0, lambda x: PI,
                                           args=(nmax[x], wavelength[x], izaDeg[x], iaaDeg[x], alphaDeg[x], betaDeg[x],
                                                 i, 1))[0]

    return QS_view

# ----------------------------------------------------------------------------------------------------------------------
# Scattering and Asymetriy Cross Section for AVERAGED FIXED ORIENTATION
# ----------------------------------------------------------------------------------------------------------------------
# Auxiliary Functions --------------------------------------------------------------------------------------------------
# Averaged Fixed Orientation
cdef double IFUNC_XSEC_QS_AF(double vza, double vaa, int nmax, double wavelength, double izaDeg, double iaaDeg,
                            int n_alpha, int n_beta, or_pdf, int i, int cos_T_sin_t):
    """Integrate Z in Viewing Direction
    
    Callable function for scipy.integrate.dblquad for integration of the phase matrix in viewing direction.

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
    i : int
        Number between 0 and 15 which defines the matrix element that will be integrated.
    cos_T_sin_t : int
        True for the calculation of the asymetry factor. This adds another term as sin(vza) to the end of the 
        integration.
        
    Returns
    -------
    Z : double
        Integrated phase matrix element.
    """
    cdef:
        double vzaDeg, vaaDeg, cos_t0, sin_t0, multi
        np.ndarray[double, ndim=2]  Z
        np.ndarray[DTYPE_t, ndim=2]  S
        np.ndarray[double, ndim=1]  Zf, I

    vzaDeg, vaaDeg = vza * RAD_TO_DEG, vaa * RAD_TO_DEG

    S, Z = SZ_AF(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, n_alpha, n_beta, or_pdf)

    Zf = Z.flatten()

    I = np.zeros(2, dtype=np.double)

    I[0] = Zf[0] + Zf[1]
    I[1] = Zf[0] - Zf[1]

    if cos_T_sin_t:
        cos_t0 = np.cos(izaDeg * DEG_TO_RAD)
        sin_t0 = np.sin(izaDeg * DEG_TO_RAD)

        multi = 0.5 * (np.sin(2 * vza) * cos_t0 +
                       (1 - np.cos(2 * vza)) * sin_t0 * np.cos((iaaDeg * DEG_TO_RAD) - vaa))

    else:
        multi = sin(vza)

    return I[i] * multi

# Scattering and Asymmetry for Averaged Fixed Orientation --------------------------------------------------------------
cdef double[:,:] XSEC_QS_AF(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg, int n_alpha,
                            int n_beta, or_pdf):
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

    cdef:
        int i
        Py_ssize_t x
        Py_ssize_t xmax = nmax.shape[0]

    QS = np.zeros((xmax, 2), dtype=np.double)
    cdef double[:,:] QS_view = QS

    for x in range(xmax):
        for i in range(2):
            QS_view[x, i] = dblquad(IFUNC_XSEC_QS_AF, 0, 2 * PI, lambda x: 0, lambda x: PI,
                                           args=(nmax[x], wavelength[x], izaDeg[x], iaaDeg[x], n_alpha, n_beta, or_pdf,
                                                 i, 0))[0]

    return QS_view

cdef double[:,:] XSEC_ASY_AF(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg, int n_alpha,
                             int n_beta, or_pdf):
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
    Qas : double[:,:]
        Two dimensional asymmetry cross section.
    """

    cdef:
        int i
        Py_ssize_t x
        Py_ssize_t xmax = nmax.shape[0]

    QS = np.zeros((xmax, 2), dtype=np.double)
    cdef double[:,:] QS_view = QS

    for x in range(xmax):
        for i in range(2):
            QS_view[x, i] = dblquad(IFUNC_XSEC_QS_AF, 0, 2 * PI, lambda x: 0, lambda x: PI,
                                           args=(nmax[x], wavelength[x], izaDeg[x], iaaDeg[x], n_alpha, n_beta, or_pdf,
                                                 i, 1))[0]

    return QS_view

# ----------------------------------------------------------------------------------------------------------------------
# Extinction Cross Section.
# ----------------------------------------------------------------------------------------------------------------------
cdef double[:,:] XSEC_QE(double complex[:,:,:] S, double[:] wavelength):
    """Extinction Cross Section.

    Parameters
    ----------
    S : double complex[:,:,:]
        Three dimensoinal scattering matrix.
    wavelength : double[:]
        Wavelength in same unit as radius (used by function calc_nmax).
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

    for x in range(xmax):
        QE[x, 0] = 2 * wavelength[x] * S[x][0, 0].imag
        QE[x, 1] = 2 * wavelength[x] * S[x][1, 1].imag

    return QE

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

cdef double[:,:] KT(double[:,:,:] ke):
    cdef:
        Py_ssize_t x, i
        Py_ssize_t xmax = ke.shape[0]

    kt = np.zeros((xmax, 2), dtype=np.double)
    cdef double[:,:] kt_view = kt

    for x in range(xmax):
        for i in range(2):
            kt_view[x, i] = 1-ke[x, i, i]

    return kt

cdef double[:,:] KA(double[:,:] ks, double[:,:] omega):
    cdef:
        Py_ssize_t x, i
        Py_ssize_t xmax = omega.shape[0]

    ka = np.zeros((xmax, 2), dtype=np.double)
    cdef double[:,:] ka_view = ka

    for x in range(xmax):
        for i in range(2):
            ka_view[x, i] = (ks[x, i] - omega[x, i] * ks[x, i]) / omega[x, i]

    return ka

cdef double[:,:] KS(double[:,:,:] ke, double[:,:] omega):
    cdef:
        Py_ssize_t x, i
        Py_ssize_t xmax = omega.shape[0]

    ks = np.zeros((xmax, 2), dtype=np.double)
    cdef double[:,:] ks_view = ks

    for x in range(xmax):
        for i in range(2):
            ks_view[x, i] = omega[x, i] * ke[x, i, i]

    return ks

cdef double[:,:,:] KE(double complex[:] factor, double complex[:,:,:] S):
    """
    Compute the extinction coefficient
    
    Parameters
    ----------
    factor : double complex[:]
        The factor to compute Mpq. The factor is (i*2*PI*N/k0)
    S : double complex[:,:,:]
        Scattering matrix.

    Returns
    -------
    KE : MemoryView, double[:,:,:]
    """
    cdef:
        Py_ssize_t x, i, j
        Py_ssize_t xmax = S.shape[0]
        double complex F, SF

    ke = np.zeros((xmax, 4, 4))
    cdef double[:,:,:] ke_view = ke

    Mpq = np.zeros((xmax, 2, 2), dtype=np.complex)
    cdef double complex[:,:,:] Mpq_view = Mpq

    for x in range(xmax):
        F = factor[x]
        for i in range(2):
            for j in range(2):
                SF = S[x, i, j]
                Mpq[x, i, j] = F * SF

    for x in range(xmax):
        ke_view[x, 0, 0] = -2 * Mpq[x, 0, 0].real
        ke_view[x, 0, 1] = 0
        ke_view[x, 0, 2] = -Mpq[x, 0, 1].real
        ke_view[x, 0, 3] = -Mpq[x, 0, 0].imag

        ke_view[x, 1, 0] = 0
        ke_view[x, 1, 1] = -2 * Mpq[x, 1, 1].real
        ke_view[x, 1, 2] = -Mpq[x, 1, 0].real
        ke_view[x, 1, 3] = -Mpq[x, 1, 0].imag

        ke_view[x, 2, 0] = -2 * Mpq[x, 1, 0].real
        ke_view[x, 2, 1] = -2 * Mpq[x, 0, 1].real
        ke_view[x, 2, 2] = -(Mpq[x, 0, 0].real + Mpq[x, 1, 1].real)
        ke_view[x, 2, 3] = Mpq[x, 0, 0].imag - Mpq[x, 1, 1].imag

        ke_view[x, 3, 0] = 2 * Mpq[x, 1, 0].imag
        ke_view[x, 3, 1] = -2 * Mpq[x, 0, 1].imag
        ke_view[x, 3, 2] = -(Mpq[x, 0, 0].imag - Mpq[x, 1, 1].imag)
        ke_view[x, 3, 3] = -(Mpq[x, 0, 0].real + Mpq[x, 1, 1].real)

    return ke

# ----------------------------------------------------------------------------------------------------------------------
# Orientation Function
# ----------------------------------------------------------------------------------------------------------------------
cdef gaussian(float std, float mean):
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
    cdef float norm_const = 1.0

    def pdf(x):
        return norm_const * np.exp(-0.5 * ((x - mean) / std) ** 2) * \
               np.sin(PI / 180.0 * x)

    cdef float norm_dev = quad(pdf, 0.0, 180.0)[0]
    # ensure that the integral over the distribution equals 1
    norm_const /= norm_dev

    return pdf

cdef uniform():
    """Uniform probability distribution function (PDF) for orientation averaging.

    Returns
    -------
    pdf(x): callable
        A function that returns the value of the spherical Jacobian-normalized uniform PDF. It is normalized for
        the interval [0, 180].
    """
    cdef float norm_const = 1.0

    def pdf(x):
        return norm_const * np.sin(PI / 180.0 * x)

    cdef float norm_dev = quad(pdf, 0.0, 180.0)[0]
    # ensure that the integral over the distribution equals 1
    norm_const /= norm_dev
    return pdf

# ----------------------------------------------------------------------------------------------------------------------
# Other Auxiliary Functions
# ----------------------------------------------------------------------------------------------------------------------
cdef double[:] equal_volume_from_maximum(double[:] radius, double[:] axis_ratio, int shape):
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
    cdef float r_eq
    cdef:
        Py_ssize_t xmax = radius.shape[0]
        Py_ssize_t x

    result = np.zeros_like(radius, dtype=np.double)
    cdef double[:] result_view = result

    for x in range(xmax):
        if shape == -1:
            if axis_ratio[x] > 1.0:  # oblate
                result_view[x] = radius[x] / axis_ratio[x] ** (1.0 / 3.0)
            else:  # prolate
                result_view[x] = radius[x] / axis_ratio[x] ** (2.0 / 3.0)
        elif shape == -2:
            if axis_ratio[x] > 1.0:  # oblate
                result_view[x] = radius[x] * (0.75 / axis_ratio[x]) ** (1.0 / 3.0)
            else:  # prolate
                result_view[x] = radius[x] * (0.75 / axis_ratio[x]) ** (2.0 / 3.0)
        else:
            raise ValueError("Parameter shape must be -1 or -2.")

        return result

# ----------------------------------------------------------------------------------------------------------------------
# T-Matrix with PSD Integrator
# ----------------------------------------------------------------------------------------------------------------------
# def init_SZ(double[:] wavelength, double[:] izaDeg, double[:] vzaDeg, double[:] iaaDeg, double[:] vaaDeg, double[:] alphaDeg,
#             double[:] betaDeg, double[:] eps_real, double[:] eps_imag, double[:] axis_ratio, int shape,
#             int radius_type, int num_points, double[:] psd_D,  int angular_integration, int verbose):
#
#     xmax = psd_D.shape[0]
#
#     for x in range(xmax):
#         D = np.zeros_like(wavelength, dtype=np.double) + psd_D[x]
#
#         if verbose == 1:
#             print("Computing point {x} at radius {D}...".format(x=x, D=D))
#
#         radius = D / 2.0
#
#         nmax = NMAX_VEC(radius=radius, radius_type=radius_type, wavelength=wavelength, eps_real=eps_real,
#                         eps_imag=eps_imag, axis_ratio=axis_ratio, shape=shape)
#
#         S, Z = SZ_S_VEC(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)
#
#         if angular_integration == 1:
#             QAS = XSEC_ASY_S(nmax, wavelength, izaDeg, iaaDeg, alphaDeg, betaDeg)
#             QS = XSEC_QS_S(nmax, wavelength, izaDeg, iaaDeg, alphaDeg, betaDeg)
#             QE = XSEC_QE(S, wavelength)

# ----------------------------------------------------------------------------------------------------------------------
# Orientation Wrapper
# ----------------------------------------------------------------------------------------------------------------------
def gaussian_wrapper(float std, float mean):
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
    return gaussian(std, mean)

def uniform_wrapper():
    """Uniform probability distribution function (PDF) for orientation averaging.

    Returns
    -------
    pdf(x): callable
        A function that returns the value of the spherical Jacobian-normalized uniform PDF. It is normalized for
        the interval [0, 180].
    """
    return uniform()

# ----------------------------------------------------------------------------------------------------------------------
# Auxiliary Wrapper
# ----------------------------------------------------------------------------------------------------------------------
def equal_volume_from_maximum_wrapper(double[:] radius, double[:] axis_ratio, int shape):
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
# SZ and NMAX Wrapper
# ----------------------------------------------------------------------------------------------------------------------
# NMAX -----------------------------------------------------------------------------------------------------------------
def NMAX_VEC_WRAPPER(double[:] radius, int radius_type, double[:] wavelength, double[:] eps_real, double[:] eps_imag,
                   double[:] axis_ratio, int shape):
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
                    axis_ratio=axis_ratio, shape=shape)

def NMAX_WRAPPER(double radius, int radius_type, double wavelength, double eps_real, double eps_imag,
                 double axis_ratio, int shape):
    """
    Calculate NMAX.

    Parameters
    ----------
    radius : double
        Equivalent particle radius in same unit as wavelength.
    radius_type : int, {0, 1, 2}
        Specification of particle radius:
            * 0: radius is the equivalent volume radius (default).
            * 1: radius is the equivalent area radius.
            * 2: radius is the maximum radius.
    wavelength : double
        Wavelength in same unit as radius.
    eps_real, eps_imag : double
        Real and imaginary part of the dielectric constant.
    axis_ratio : double
        The horizontal-to-rotational axis ratio.
    shape : int, {-1, -2}
        Shape of the particle:
            * -1 : spheroid,
            * -2 : cylinders.

    Returns
    -------
    nmax : int

    """

    return NMAX(radius=radius, radius_type=radius_type, wavelength=wavelength, eps_real=eps_real, eps_imag=eps_imag,
                axis_ratio=axis_ratio, shape=shape)

# S and Z for Single Orientation ---------------------------------------------------------------------------------------
def SZ_S_VEC_WRAPPER(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] vzaDeg, double[:] iaaDeg,
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

def SZ_S_WRAPPER(int nmax, double wavelength, double izaDeg, double vzaDeg, double iaaDeg,
                 double vaaDeg, double alphaDeg, double betaDeg):
    """
    Calculate the single scattering and phase matrix.

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
    return SZ_S(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)

# S and Z for Adaptive Fixed Orientation -------------------------------------------------------------------------------
def SZ_AF_VEC_WRAPPER(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] vzaDeg, double[:] iaaDeg,
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


def SZ_AF_WRAPPER(int nmax, double wavelength, double izaDeg, double vzaDeg, double iaaDeg,
                  double vaaDeg, int n_alpha, int n_beta, or_pdf):
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
        Two dimensional scattering (S) and phase (Z) matrix.
    """

    return SZ_AF(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, n_alpha, n_beta, or_pdf)

# ----------------------------------------------------------------------------------------------------------------------
# Integrate Scattering and Phase Matrix Wrapper
# ----------------------------------------------------------------------------------------------------------------------
# S and Z for Single Orientation ---------------------------------------------------------------------------------------
def DBLQUAD_Z_S_WRAPPER(int[:] nmax, double[:] wavelength, double[:] xzaDeg, double[:] xaaDeg, double[:] alphaDeg,
                        double[:] betaDeg, int iza_flag):
    """
    Integrate the phase matrix over a half sphere for SINGLE ORIENTATION.

    Parameters
    ----------
    nmax : int[:]
        Nmax parameter.
    wavelength : double[:]
        Wavelength in same unit as radius (used by function calc_nmax).
    xzaDeg, xaaDeg : double[:]
        Scattering (vza) or incidence (iza) zenith and azimuth angle (vaa, iaa) in [DEG].
    alphaDeg, betaDeg: double[:]
        The Euler angles of the particle orientation in [DEG].
    iza_flag : int
        If True, the integration is over the incidence angle. Then the xzaDeg and xaaDeg must be the viewing angle
        vzaDeg and vaaDeg. Else it will be integrated over the viewing direction where xzaDeg and xaaDeg must be the
        incidence angle izaDeg and iaaDeg.
    Returns
    -------
    Z : MemoryView, double[:,:]
        Two dimensional integrated phase matrix.

    Note
    ----
    The output is a array with shape (nmax.shape[0], 16). To put it in a matrix form one must reshape it like
    np.reshape(Z, (nmax.shape[0], 4, 4)).
    """

    if iza_flag:
        return DBLQUAD_Z_S_IZA(nmax, wavelength, xzaDeg, xaaDeg, alphaDeg, betaDeg)
    else:
        return DBLQUAD_Z_S_VZA(nmax, wavelength, xzaDeg, xaaDeg, alphaDeg, betaDeg)

# S and Z for Adaptive Fixed Orientation -------------------------------------------------------------------------------
def DBLQUAD_Z_AF_WRAPPER(int[:] nmax, double[:] wavelength, double[:] xzaDeg, double[:] xaaDeg, int n_alpha, int n_beta,
                         or_pdf, int iza_flag):
    """
    Integrate the phase matrix over a half sphere for Adaptive Fixed Orientation.

    Parameters
    ----------
    nmax : int[:]
        Nmax parameter.
    wavelength : double[:]
        Wavelength in same unit as radius (used by function calc_nmax).
    xzaDeg, xaaDeg : double[:]
        Scattering (vza) or incidence (iza) zenith and azimuth angle (vaa, iaa) in [DEG].
    alphaDeg, betaDeg: double[:]
        The Euler angles of the particle orientation in [DEG].
    iza_flag : int
        If True, the integration is over the incidence angle. Then the xzaDeg and xaaDeg must be the viewing angle
        vzaDeg and vaaDeg. Else it will be integrated over the viewing direction where xzaDeg and xaaDeg must be the
        incidence angle izaDeg and iaaDeg.
    Returns
    -------
    Z : MemoryView, double[:,:]
        Two dimensional integrated phase matrix.

    Note
    ----
    The output is a array with shape (nmax.shape[0], 16). To put it in a matrix form one must reshape it like
    np.reshape(Z, (nmax.shape[0], 4, 4)).
    """

    if iza_flag:
        return DBLQUAD_Z_AF_IZA(nmax, wavelength, xzaDeg, xaaDeg, n_alpha, n_beta, or_pdf)
    else:
        return DBLQUAD_Z_AF_VZA(nmax, wavelength, xzaDeg, xaaDeg, n_alpha, n_beta, or_pdf)

# ----------------------------------------------------------------------------------------------------------------------
# Cross Section Wrapper
# ----------------------------------------------------------------------------------------------------------------------
# Single Orientation ---------------------------------------------------------------------------------------------------
def XSEC_QS_S_WRAPPER(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg, double[:] alphaDeg,
                      double[:] betaDeg):
    """Scattering Cross Section for single orientation.

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
    Qs : double[:,:]
        Two dimensional scattering cross section.
    """

    return XSEC_QS_S(nmax, wavelength, izaDeg, iaaDeg, alphaDeg,betaDeg)

def XSEC_ASY_S_WRAPPER(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg, double[:] alphaDeg,
                       double[:] betaDeg):
    """Asymetry Cross Section for single orientation.

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
    Qas : double[:,:]
        Two dimensional asymmetry cross section.

    Note
    ----
    To obtain the asymetry factor the output of this function must be divided through the scattering cross section.
    """

    return XSEC_ASY_S(nmax, wavelength, izaDeg, iaaDeg, alphaDeg, betaDeg)

# Averaged Fixed Orientation -------------------------------------------------------------------------------------------
def XSEC_QS_AF_WRAPPER(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg, int n_alpha, int n_beta,
                         or_pdf):
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

    return XSEC_QS_AF(nmax, wavelength, izaDeg, iaaDeg, n_alpha, n_beta, or_pdf)

def XSEC_ASY_AF_WRAPPER(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg, int n_alpha, int n_beta,
                         or_pdf):
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
    Qas : double[:,:]
        Two dimensional asymmetry cross section.
    """

    return XSEC_ASY_AF(nmax, wavelength, izaDeg, iaaDeg, n_alpha, n_beta, or_pdf)

# Extinction and Intensity ---------------------------------------------------------------------------------------------
def XSEC_QE_WRAPPER(double complex[:,:,:] S, double[:] wavelength):
    """Extinction Cross Section.

    Parameters
    ----------
    S : double complex[:,:,:]
        Three dimensoinal scattering matrix.
    wavelength : double[:]
        Wavelength in same unit as radius (used by function calc_nmax).
    Returns
    -------
    Qe : double[:,:]
        Two dimensional extinction cross section.
    """
    return XSEC_QE(S, wavelength)

def XSEC_QSI_WRAPPER(double[:,:,:] Z):
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

def KE_WRAPPER(double complex[:] factor, double complex[:,:,:] S):
    return KE(factor, S)

def KS_WRAPPER(double[:,:,:] ke, double[:,:] omega):
    return KS(ke, omega)

def KA_WRAPPER(double[:,:] ks, double[:,:] omega):
    return KA(ks, omega)

def KT_WRAPPER(double[:,:,:] ke):
    return KT(ke)
# ----------------------------------------------------------------------------------------------------------------------
# Not Used Functions Yet
# ----------------------------------------------------------------------------------------------------------------------
# def orient_averaged_adaptive_wrapper(float nmax, float wavelength, float izaDeg, float vzaDeg, float iaaDeg,
#                                      float vaaDeg, or_pdf):
#     return orient_averaged_adaptive(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, or_pdf)

# cdef orient_averaged_adaptive(float nmax, float wavelength, float izaDeg, float vzaDeg, float iaaDeg, float vaaDeg,
#                               or_pdf):
#     """Compute the T-matrix using variable orientation scatterers.
#
#     This method uses a very slow adaptive routine and should mainly be used
#     for reference purposes. Uses the set particle orientation PDF, ignoring
#     the alpha and beta attributes.
#
#     Args:
#         : selfatrix (or descendant) instance
#
#     Returns:
#         The amplitude (S) and phase (Z) matrices.
#     """
#     cdef np.ndarray S = np.zeros((2, 2), dtype=complex)
#     cdef np.ndarray Z = np.zeros((4, 4))
#     cdef int i, j
#
#     ind = range(2)
#     for i in ind:
#         for j in ind:
#             S.real[i, j] = dblquad(ifunc_S, 0.0, 360.0,
#                                    lambda x: 0.0, lambda x: 180.0,
#                                    (i, j, 1, nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, or_pdf))[0] / 360.0
#             S.imag[i, j] = dblquad(ifunc_S, 0.0, 360.0,
#                                    lambda x: 0.0, lambda x: 180.0,
#                                    (i, j, 0, nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, or_pdf))[0] / 360.0
#
#     ind = range(4)
#     for i in ind:
#         for j in ind:
#             Z[i, j] = dblquad(ifunc_Z, 0.0, 360.0,
#                               lambda x: 0.0, lambda x: 180.0,
#                               (i, j, 1, nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, or_pdf))[0] / 360.0
#
#     return S, Z
