# cython: cdivision=True
from __future__ import division

cimport numpy as np
import numpy as np
from libc.math cimport pow, pi, abs, cos
import cmath
from scipy.integrate import quad as squad

DTYPE = np.float
ctypedef np.float_t DTYPE_t

DTYPEC = np.complex128
ctypedef np.complex128_t DTYPEC_t

cdef float EPSILON = 2.220446049250313e-16

def is_forward_angle(DTYPEC_t n, DTYPEC_t theta):
    """
    if a wave is traveling at angle theta from normal in a medium with index n,
    calculate whether or not this is the forward-traveling wave (i.e., the one
    going from front to back of the stack, like the incoming or outgoing waves,
    but unlike the reflected wave). For real n & theta, the criterion is simply
    -pi/2 < theta < pi/2, but for complex n & theta, it's more complicated.
    See https://arxiv.org/abs/1603.02720 appendix D. If theta is the forward
    angle, then (pi-theta) is the backward angle and vice-versa.
    """

    assert n.real * n.imag >= 0, ("For materials with gain, it's ambiguous which "
                                  "beam is incoming vs outgoing. See "
                                  "https://arxiv.org/abs/1603.02720 Appendix C.\n"
                                  "n: " + str(n) + "   angle: " + str(theta))

    cdef DTYPEC_t ncostheta = n * cmath.cos(theta)

    if abs(ncostheta.imag) > 100 * EPSILON:
        # Either evanescent decay or lossy medium. Either way, the one that
        # decays is the forward-moving wave
        answer = (ncostheta.imag > 0)
    else:
        # Forward is the one with positive Poynting vector
        # Poynting vector is Re[n cos(theta)] for s-polarization or
        # Re[n cos(theta*)] for p-polarization, but it turns out they're consistent
        # so I'll just assume s then check both below
        answer = (ncostheta.real > 0)
    # convert from numpy boolean to the normal Python boolean
    answer = bool(answer)
    # double-check the answer ... can't be too careful!
    error_string = ("It's not clear which beam is incoming vs outgoing. Weird"
                    " index maybe?\n"
                    "n: " + str(n) + "   angle: " + str(theta))

    if answer is True:
        assert ncostheta.imag > -100 * EPSILON, error_string
        assert ncostheta.real > -100 * EPSILON, error_string
        assert (n * cmath.cos(np.conjugate(theta))).real > -100 * EPSILON, error_string
    else:
        assert ncostheta.imag < 100 * EPSILON, error_string
        assert ncostheta.real < 100 * EPSILON, error_string
        assert (n * cmath.cos(np.conjugate(theta))).real < 100 * EPSILON, error_string
    return answer

cdef snell_c(DTYPE_t iza, DTYPEC_t n1, DTYPEC_t n2):
    """
    return angle theta in layer 2 with refractive index n_2, assuming
    it has angle th_1 in layer with refractive index n_1. Use Snell's law. Note
    that "angles" may be complex!!
    """

    cdef DTYPEC_t th_2_guess = cmath.asin(n1 * np.sin(iza) / n2)

    try:
        if is_forward_angle(n2, th_2_guess):

            return th_2_guess
        else:
            return pi - th_2_guess

    except AssertionError:
        return th_2_guess

cdef reflection_c(DTYPE_t xza, DTYPEC_t n1, DTYPEC_t n2):
    """
    reflection amplitude (from Fresnel equations)
    polarization is either "s" or "p" for polarization
    n_i, n_f are (complex) refractive index for incident and final
    th_i, th_f are (complex) propegation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".
    """

    cdef DTYPEC_t rza = snell_c(xza, n1, n2)

    cdef DTYPEC_t V = ((n2 * cos(xza) - n1 * cmath.cos(rza)) /
                       (n2 * cos(xza) + n1 * cmath.cos(rza)))

    cdef DTYPEC_t H = ((n1 * cos(xza) - n2 * cmath.cos(rza)) /
                       (n1 * cos(xza) + n2 * cmath.cos(rza)))

    return V, H

def reflection_coefficients(float iza, double complex eps):
    cdef double complex rt = np.sqrt(eps - pow(np.sin(iza + 0.01), 2))
    cdef double complex Rvi = (eps * np.cos(iza + 0.01) - rt) / (eps * np.cos(iza + 0.01) + rt)
    cdef double complex Rhi = (np.cos(iza + 0.01) - rt) / (np.cos(iza + 0.01) + rt)

    return Rvi, Rhi

cdef transmission_c(DTYPE_t xza, DTYPEC_t n1, DTYPEC_t n2):
    """
    transmission amplitude (frem Fresnel equations)
    polarization is either "s" or "p" for polarization
    n_i, n_f are (complex) refractive index for incident and final
    th_i, th_f are (complex) propegation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".
    """

    cdef DTYPEC_t rza = snell_c(xza, n1, n2)

    cdef DTYPEC_t V = 2 * n2 * cos(xza) / (n2 * cos(xza) + n1 * cmath.cos(rza))
    cdef DTYPEC_t H = 2 * n2 * cos(xza) / (n1 * cos(xza) + n2 * cmath.cos(rza))

    return V, H

cdef r11c(float xza, double complex eps, _):
    V, H = reflection_coefficients(xza, eps)

    return pow(abs(V), 2)

cdef r22c(float xza, double complex eps, _):
    V, H = reflection_coefficients(xza, eps)

    return pow(abs(H), 2)

cdef r33c(float xza, double complex eps, _):
    V, H = reflection_coefficients(xza, eps)

    return np.real(V * np.conjugate(H))

cdef r34c(float xza, double complex eps, _):
    V, H = reflection_coefficients(xza, eps)

    return -np.imag(V * np.conjugate(H))

cdef r43c(float xza, double complex eps, _):
    V, H = reflection_coefficients(xza, eps)

    return np.imag(V * np.conjugate(H))

cdef r44c(float xza, double complex eps, _):
    V, H = reflection_coefficients(xza, eps)

    return np.real(V * np.conjugate(H))

cdef reflectivity_c(float xza, double complex eps):
    cdef np.ndarray[DTYPE_t, ndim=2] mat = np.zeros((4, 4), dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=2] mat2 = np.zeros((5, 5), dtype=np.float)

    mat[0, 0] = r11c(xza, eps, 0)
    mat[1, 1] = r22c(xza, eps, 0)
    mat[2, 2] = r33c(xza, eps, 0)
    mat[2, 3] = r34c(xza, eps, 0)
    mat[3, 2] = r43c(xza, eps, 0)
    mat[3, 3] = r44c(xza, eps, 0)

    return mat

# cdef transmissivity_c(DTYPE_t xza, DTYPEC_t n1, DTYPEC_t n2):
#     V, H = transmission_c(xza, n1, n2)
#
#     cdef np.ndarray[DTYPE_t, ndim=2] mat = np.zeros((4, 4), dtype=np.float)
#     cdef np.ndarray[DTYPE_t, ndim=2] mat2 = np.zeros((5, 5), dtype=np.float)
#
#     cdef DTYPEC_t rza = snell_c(xza, n1, n2)
#
#     factor = ((n1 * cmath.cos(rza)).real / (n2 * cos(xza)).real)
#
#
#     mat[0, 0] = r11c(xza, n1, n2)
#     mat[1, 1] = r22c(xza, n1, n2)
#     mat[2, 2] = r33c(xza, n1, n2)
#     mat[2, 3] = r34c(xza, n1, n2)
#     mat[3, 2] = r43c(xza, n1, n2)
#     mat[3, 3] = r44c(xza, n1, n2)
#
#     return mat * factor


cdef quad(float a, float b, double complex eps):
    cdef np.ndarray[DTYPE_t, ndim=2] mat = np.zeros((4, 4), dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=2] mat2 = np.zeros((5, 5), dtype=np.float)

    r11ci = squad(r11c, a, b, args=(eps, 0))[0]
    r22ci = squad(r22c, a, b, args=(eps, 0))[0]
    r33ci = squad(r33c, a, b, args=(eps, 0))[0]
    r34ci = squad(r34c, a, b, args=(eps, 0))[0]
    r43ci = squad(r43c, a, b, args=(eps, 0))[0]
    r44ci = squad(r44c, a, b, args=(eps, 0))[0]

    mat[0, 0] = r11ci
    mat[1, 1] = r22ci
    mat[2, 2] = r33ci
    mat[2, 3] = r34ci
    mat[3, 2] = r43ci
    mat[3, 3] = r44ci

    return mat

def r11(float xza, double complex eps, _):
    return r11c(xza, eps, _)

def r22(float xza, double complex eps, _):
    return r22c(xza, eps, _)

def r33(float xza, double complex eps, _):
    return r33c(xza, eps, _)

def r34(float xza, double complex eps, _):
    return r34c(xza, eps, _)

def r43(float xza, double complex eps, _):
    return r43c(xza, eps, _)

def r44(float xza, double complex eps, _):
    return r44c(xza, eps, _)

def reflectivity_wrapper(float xza, double complex eps):
    return reflectivity_c(xza, eps)

# def transmissivity_wrapper(DTYPE_t xza, DTYPEC_t n1, DTYPEC_t n2):
#     return transmissivity_c(xza, n1, n2)

def snell_wrapper(DTYPE_t iza, DTYPEC_t n1, DTYPEC_t n2):
    return snell_c(iza, n1, n2)

def quad_wrapper(float a, float b, double complex eps):
    return quad(a, b, eps)
