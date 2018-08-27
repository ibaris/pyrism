# cython: cdivision=True
from __future__ import division

cimport numpy as np
import numpy as np
from libc.math cimport pow, pi, sqrt, abs, asin, cos
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

cdef r00(DTYPE_t xza, DTYPEC_t n1, DTYPEC_t n2):
    V, H = reflection_c(xza, n1, n2)

    cdef DTYPE_t iso = (r11(xza, n1, n2) + r22(xza, n1, n2) + r33(xza, n1, n2) + r34(xza, n1, n2) + r43(xza, n1,
                                                                                                        n2) + r44(xza,
                                                                                                                  n1,
                                                                                                                  n2)) / 4

    return iso

cdef r11(DTYPE_t xza, DTYPEC_t n1, DTYPEC_t n2):
    V, H = reflection_c(xza, n1, n2)

    return pow(abs(V), 2)

cdef r22(DTYPE_t xza, DTYPEC_t n1, DTYPEC_t n2):
    V, H = reflection_c(xza, n1, n2)

    return pow(abs(H), 2)

cdef r33(DTYPE_t xza, DTYPEC_t n1, DTYPEC_t n2):
    V, H = reflection_c(xza, n1, n2)

    return np.real(V * np.conjugate(H))

cdef r34(DTYPE_t xza, DTYPEC_t n1, DTYPEC_t n2):
    V, H = reflection_c(xza, n1, n2)

    return -np.imag(V * np.conjugate(H))

cdef r43(DTYPE_t xza, DTYPEC_t n1, DTYPEC_t n2):
    V, H = reflection_c(xza, n1, n2)

    return np.imag(V * np.conjugate(H))

cdef r44(DTYPE_t xza, DTYPEC_t n1, DTYPEC_t n2):
    V, H = reflection_c(xza, n1, n2)

    return np.real(V * np.conjugate(H))

cdef reflectivity_c(DTYPE_t xza, DTYPEC_t n1, DTYPEC_t n2, int iso):
    cdef np.ndarray[DTYPE_t, ndim=2] mat = np.zeros((4, 4), dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=2] mat2 = np.zeros((5, 5), dtype=np.float)

    V, H = reflection_c(xza, n1, n2)

    if iso == 0:

        mat[0, 0] = r11(xza, n1, n2)
        mat[1, 1] = r22(xza, n1, n2)
        mat[2, 2] = r33(xza, n1, n2)
        mat[2, 3] = r34(xza, n1, n2)
        mat[3, 2] = r43(xza, n1, n2)
        mat[3, 3] = r44(xza, n1, n2)

        return mat

    else:

        mat2[0, 0] = r00(xza, n1, n2)
        mat2[1, 1] = r11(xza, n1, n2)
        mat2[2, 2] = r22(xza, n1, n2)
        mat2[3, 3] = r33(xza, n1, n2)
        mat2[3, 4] = r34(xza, n1, n2)
        mat2[4, 3] = r43(xza, n1, n2)
        mat2[4, 4] = r44(xza, n1, n2)

        return mat2

cdef transmissivity_c(DTYPE_t xza, DTYPEC_t n1, DTYPEC_t n2, int iso):
    V, H = transmission_c(xza, n1, n2)

    cdef np.ndarray[DTYPE_t, ndim=2] mat = np.zeros((4, 4), dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=2] mat2 = np.zeros((5, 5), dtype=np.float)

    cdef DTYPEC_t rza = snell_c(xza, n1, n2)

    factor = ((n1 * cmath.cos(rza)).real / (n2 * cos(xza)).real)

    if iso == 0:

        mat[0, 0] = r11(xza, n1, n2)
        mat[1, 1] = r22(xza, n1, n2)
        mat[2, 2] = r33(xza, n1, n2)
        mat[2, 3] = r34(xza, n1, n2)
        mat[3, 2] = r43(xza, n1, n2)
        mat[3, 3] = r44(xza, n1, n2)

        return mat * factor

    else:

        mat2[0, 0] = r00(xza, n1, n2)
        mat2[1, 1] = r11(xza, n1, n2)
        mat2[2, 2] = r22(xza, n1, n2)
        mat2[3, 3] = r33(xza, n1, n2)
        mat2[3, 4] = r34(xza, n1, n2)
        mat2[4, 3] = r43(xza, n1, n2)
        mat2[4, 4] = r44(xza, n1, n2)

        return mat2 * factor

cdef quad(float a, float b, DTYPEC_t n1, DTYPEC_t n2, int iso):
    cdef np.ndarray[DTYPE_t, ndim=2] mat = np.zeros((4, 4), dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=2] mat2 = np.zeros((5, 5), dtype=np.float)

    if iso == 0:
        r11i = squad(r11, a, b, args=(n1, n2))[0]
        r22i = squad(r22, a, b, args=(n1, n2))[0]
        r33i = squad(r33, a, b, args=(n1, n2))[0]
        r34i = squad(r34, a, b, args=(n1, n2))[0]
        r43i = squad(r43, a, b, args=(n1, n2))[0]
        r44i = squad(r44, a, b, args=(n1, n2))[0]

        mat[0, 0] = r11i
        mat[1, 1] = r22i
        mat[2, 2] = r33i
        mat[2, 3] = r34i
        mat[3, 2] = r43i
        mat[3, 3] = r44i

        return mat

    else:
        r00i = squad(r00, a, b, args=(n1, n2))[0]
        r11i = squad(r11, a, b, args=(n1, n2))[0]
        r22i = squad(r22, a, b, args=(n1, n2))[0]
        r33i = squad(r33, a, b, args=(n1, n2))[0]
        r34i = squad(r34, a, b, args=(n1, n2))[0]
        r43i = squad(r43, a, b, args=(n1, n2))[0]
        r44i = squad(r44, a, b, args=(n1, n2))[0]

        mat2[0, 0] = r00i
        mat2[1, 1] = r11i
        mat2[2, 2] = r22i
        mat2[3, 3] = r33i
        mat2[3, 4] = r34i
        mat2[4, 3] = r43i
        mat2[4, 4] = r44i

        return mat2

def reflectivity_wrapper(DTYPE_t xza, DTYPEC_t n1, DTYPEC_t n2, int iso):
    return reflectivity_c(xza, n1, n2, iso)

def transmissivity_wrapper(DTYPE_t xza, DTYPEC_t n1, DTYPEC_t n2, int iso):
    return transmissivity_c(xza, n1, n2, iso)

def snell_wrapper(DTYPE_t iza, DTYPEC_t n1, DTYPEC_t n2):
    return snell_c(iza, n1, n2)

def quad_wrapper(float a, float b, DTYPEC_t n1, DTYPEC_t n2, int iso):
    return quad(a, b, n1, n2, iso)
