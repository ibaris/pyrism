# cython: cdivision=True

cimport numpy as np
import numpy as np
from libc.math cimport cos, sin, pow, pi

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef p11_c(DTYPE_t iza, DTYPE_t vza, DTYPE_t raa):
    cdef float first = pow(sin(vza), 2) * pow(sin(iza), 2)
    cdef float second = pow(cos(vza), 2) * pow(cos(iza), 2) * pow(cos(raa), 2)
    cdef float third = 2 * sin(vza) * sin(iza) * cos(vza) * cos(iza) * cos(raa)

    return first + second + third

cdef p12_c(DTYPE_t iza, DTYPE_t vza, DTYPE_t raa):
    cdef float first = pow(cos(vza), 2) * pow(sin(raa), 2)

    return first

cdef p13_c(DTYPE_t iza, DTYPE_t vza, DTYPE_t raa):
    cdef float first = cos(vza) * sin(vza) * sin(iza) * sin(raa)
    cdef float second = pow(cos(vza), 2) * cos(iza) * sin(raa) * cos(raa)

    return first + second

cdef p21_c(DTYPE_t iza, DTYPE_t vza, DTYPE_t raa):
    cdef float first = pow(cos(iza), 2) * pow(sin(raa), 2)

    return first

cdef p22_c(DTYPE_t iza, DTYPE_t vza, DTYPE_t raa):
    cdef float first = pow(cos(raa), 2)

    return first

cdef p23_c(DTYPE_t iza, DTYPE_t vza, DTYPE_t raa):
    cdef float first = -cos(iza) * sin(raa) * cos(raa)

    return first

cdef p31_c(DTYPE_t iza, DTYPE_t vza, DTYPE_t raa):
    cdef float first = -2 * cos(vza) * sin(iza) * cos(iza) * sin(raa)
    cdef float second = - cos(vza) * pow(cos(iza), 2) * cos(raa) * sin(raa)

    return first + second

cdef p32_c(DTYPE_t iza, DTYPE_t vza, DTYPE_t raa):
    cdef float first = 2 * cos(vza) * sin(raa) * cos(raa)

    return first

cdef p33_c(DTYPE_t iza, DTYPE_t vza, DTYPE_t raa):
    cdef float first = sin(vza) * sin(iza) * cos(raa)
    cdef float second = cos(vza) * cos(iza) * cos(2 * raa)

    return first + second

cdef p44_c(DTYPE_t iza, DTYPE_t vza, DTYPE_t raa):
    cdef float first = sin(vza) * sin(iza) * cos(raa)
    cdef float second = cos(vza) * cos(iza)

    return first + second

cdef p11_int_c(DTYPE_t vza):
    return pi * sin(vza) ** 2 + pi / 3

cdef p12_int_c(DTYPE_t vza):
    return pi * cos(vza) ** 2

cdef p21_int_c(DTYPE_t vza):
    return pi / 3

cdef p22_int_c(DTYPE_t vza):
    return pi

cdef p44_int_c(DTYPE_t vza):
    return pi * cos(vza)

cdef phase_matrix_c(DTYPE_t iza, DTYPE_t vza, DTYPE_t raa):
    cdef np.ndarray[DTYPE_t, ndim=2] mat = np.zeros((4, 4), dtype=np.float)

    mat[0, 0] = p11_c(iza, vza, raa)
    mat[0, 1] = p12_c(iza, vza, raa)
    mat[0, 2] = p13_c(iza, vza, raa)

    mat[1, 0] = p21_c(iza, vza, raa)
    mat[1, 1] = p22_c(iza, vza, raa)
    mat[1, 2] = p23_c(iza, vza, raa)

    mat[2, 0] = p31_c(iza, vza, raa)
    mat[2, 1] = p32_c(iza, vza, raa)
    mat[2, 2] = p33_c(iza, vza, raa)

    mat[3, 3] = p44_c(iza, vza, raa)

    return mat

cdef phase_matrix_int_c(DTYPE_t vza):
    cdef np.ndarray[DTYPE_t, ndim=2] mat = np.zeros((4, 4), dtype=np.float)

    mat[0, 0] = p11_int_c(vza)
    mat[0, 1] = p12_int_c(vza)

    mat[1, 0] = p21_int_c(vza)
    mat[1, 1] = p22_int_c(vza)

    mat[3, 3] = p44_int_c(vza)

    return mat

def phase_matrix(iza, vza, raa, integration=0):
    if integration == 0:
        return phase_matrix_c(iza, vza, raa)
    if integration == 1:
        return phase_matrix_int_c(vza)
