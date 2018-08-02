# cython: cdivision=True

cimport numpy as np
import numpy as np
from libc.math cimport pow, pi, sqrt, abs
import cmath

cdef rayleigh_scattering_c(float frequency, float radius, double complex eps_p, double complex eps_b):
    cdef float eps_b_real = eps_b.real

    cdef double complex n_p = cmath.sqrt(eps_p)
    cdef double complex n_b = cmath.sqrt(eps_b)
    cdef double complex n = n_p / n_b

    cdef float chi = 20.0 / 3.0 * pi * radius * frequency * sqrt(eps_b_real)
    cdef double complex BigK = ((n * n) - 1) / ((n * n) + 2)

    cdef float ks = 8.0 / 3.0 * pow(chi, 4) * pow(abs(BigK), 2)
    cdef float ka = 4.0 * chi * (-BigK.imag)
    cdef float bsc = 4.0 * pow(chi, 4) * pow(abs(BigK), 2)

    cdef float ke = ka + ks
    cdef float kt = 1 - ke
    cdef float omega = ks / ke

    return ks, ka, kt, ke, omega, bsc
def rayleigh_scattering_wrapper(float frequency, float radius, double complex eps_p,
                                double complex eps_b):
    return rayleigh_scattering_c(frequency, radius, eps_p, eps_b)
