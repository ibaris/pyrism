# cython: cdivision=True
from __future__ import division

cimport numpy as np
import numpy as np
from libc.math cimport cos, sin, pow, pi
from scipy.integrate import dblquad as sdblquad
from scipy.integrate import quad as squad

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef p11_c(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza):
    cdef float first = pow(sin(vza), 2) * pow(sin(iza), 2)
    cdef float second = pow(cos(vza), 2) * pow(cos(iza), 2) * pow(cos(raa), 2)
    cdef float third = 2 * sin(vza) * sin(iza) * cos(vza) * cos(iza) * cos(raa)

    return first + second + third

cdef p12_c(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza):
    cdef float first = pow(cos(vza), 2) * pow(sin(raa), 2)

    return first

cdef p13_c(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza):
    cdef float first = cos(vza) * sin(vza) * sin(iza) * sin(raa)
    cdef float second = pow(cos(vza), 2) * cos(iza) * sin(raa) * cos(raa)

    return first + second

cdef p21_c(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza):
    cdef float first = pow(cos(iza), 2) * pow(sin(raa), 2)

    return first

cdef p22_c(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza):
    cdef float first = pow(cos(raa), 2)

    return first

cdef p23_c(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza):
    cdef float first = -cos(iza) * sin(raa) * cos(raa)

    return first

cdef p31_c(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza):
    cdef float first = -2 * cos(vza) * sin(iza) * cos(iza) * sin(raa)
    cdef float second = - cos(vza) * pow(cos(iza), 2) * cos(raa) * sin(raa)

    return first + second

cdef p32_c(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza):
    cdef float first = 2 * cos(vza) * sin(raa) * cos(raa)

    return first

cdef p33_c(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza):
    cdef float first = sin(vza) * sin(iza) * cos(raa)
    cdef float second = cos(vza) * cos(iza) * cos(2 * raa)

    return first + second

cdef p44_c(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza):
    cdef float first = sin(vza) * sin(iza) * cos(raa)
    cdef float second = cos(vza) * cos(iza)

    return first + second

def p11_c_wrapper(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza, _):
    return p11_c(iza, raa, vza) * sin(iza)

def p12_c_wrapper(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza, _):
    return p12_c(iza, raa, vza) * sin(iza)

def p13_c_wrapper(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza, _):
    return p13_c(iza, raa, vza) * sin(iza)

def p21_c_wrapper(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza, _):
    return p21_c(iza, raa, vza) * sin(iza)

def p22_c_wrapper(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza, _):
    return p22_c(iza, raa, vza) * sin(iza)

def p23_c_wrapper(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza, _):
    return p23_c(iza, raa, vza) * sin(iza)

def p31_c_wrapper(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza, _):
    return p31_c(iza, raa, vza) * sin(iza)

def p32_c_wrapper(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza, _):
    return p32_c(iza, raa, vza) * sin(iza)

def p33_c_wrapper(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza, _):
    return p33_c(iza, raa, vza) * sin(iza)

def p44_c_wrapper(DTYPE_t iza, DTYPE_t raa, DTYPE_t vza, _):
    return p44_c(iza, raa, vza) * sin(iza)

cdef pmatrix_c(DTYPE_t iza, DTYPE_t vza, DTYPE_t raa):
    cdef np.ndarray[DTYPE_t, ndim=2] mat = np.zeros((4, 4), dtype=np.float)

    mat[0, 0] = p11_c(iza, raa, vza)
    mat[0, 1] = p12_c(iza, raa, vza)
    mat[0, 2] = p13_c(iza, raa, vza)

    mat[1, 0] = p21_c(iza, raa, vza)
    mat[1, 1] = p22_c(iza, raa, vza)
    mat[1, 2] = p23_c(iza, raa, vza)

    mat[2, 0] = p31_c(iza, raa, vza)
    mat[2, 1] = p32_c(iza, raa, vza)
    mat[2, 2] = p33_c(iza, raa, vza)

    mat[3, 3] = p44_c(iza, raa, vza)

    return mat

cdef dblquad_c(DTYPE_t vza, float a, float b, float g, float h):
    cdef float p11 = sdblquad(p11_c_wrapper, a, b, lambda x: g, lambda x: h, args=(vza, 0))[0]
    cdef float p12 = sdblquad(p12_c_wrapper, a, b, lambda x: g, lambda x: h, args=(vza, 0))[0]
    cdef float p13 = sdblquad(p13_c_wrapper, a, b, lambda x: g, lambda x: h, args=(vza, 0))[0]

    cdef float p21 = sdblquad(p21_c_wrapper, a, b, lambda x: g, lambda x: h, args=(vza, 0))[0]
    cdef float p22 = sdblquad(p22_c_wrapper, a, b, lambda x: g, lambda x: h, args=(vza, 0))[0]
    cdef float p23 = sdblquad(p23_c_wrapper, a, b, lambda x: g, lambda x: h, args=(vza, 0))[0]

    cdef float p31 = sdblquad(p31_c_wrapper, a, b, lambda x: g, lambda x: h, args=(vza, 0))[0]
    cdef float p32 = sdblquad(p32_c_wrapper, a, b, lambda x: g, lambda x: h, args=(vza, 0))[0]
    cdef float p33 = sdblquad(p33_c_wrapper, a, b, lambda x: g, lambda x: h, args=(vza, 0))[0]

    cdef float p44 = sdblquad(p44_c_wrapper, a, b, lambda x: g, lambda x: h, args=(vza, 0))[0]

    cdef np.ndarray[DTYPE_t, ndim=2] mat = np.zeros((4, 4), dtype=np.float)

    mat[0, 0] = p11
    mat[0, 1] = p12
    mat[0, 2] = p13

    mat[1, 0] = p21
    mat[1, 1] = p22
    mat[1, 2] = p23

    mat[2, 0] = p31
    mat[2, 1] = p32
    mat[2, 2] = p33

    mat[3, 3] = p44

    return mat

cdef quad_c(DTYPE_t vza, DTYPE_t raa, float a, float b):
    cdef float p11 = squad(p11_c_wrapper, a, b, args=(raa, vza, 0))[0]
    cdef float p12 = squad(p12_c_wrapper, a, b, args=(raa, vza, 0))[0]
    cdef float p13 = squad(p13_c_wrapper, a, b, args=(raa, vza, 0))[0]

    cdef float p21 = squad(p21_c_wrapper, a, b, args=(raa, vza, 0))[0]
    cdef float p22 = squad(p22_c_wrapper, a, b, args=(raa, vza, 0))[0]
    cdef float p23 = squad(p23_c_wrapper, a, b, args=(raa, vza, 0))[0]

    cdef float p31 = squad(p31_c_wrapper, a, b, args=(raa, vza, 0))[0]
    cdef float p32 = squad(p32_c_wrapper, a, b, args=(raa, vza, 0))[0]
    cdef float p33 = squad(p33_c_wrapper, a, b, args=(raa, vza, 0))[0]

    cdef float p44 = squad(p44_c_wrapper, a, b, args=(raa, vza, 0))[0]

    cdef np.ndarray[DTYPE_t, ndim=2] mat = np.zeros((4, 4), dtype=np.float)

    mat[0, 0] = p11
    mat[0, 1] = p12
    mat[0, 2] = p13

    mat[1, 0] = p21
    mat[1, 1] = p22
    mat[1, 2] = p23

    mat[2, 0] = p31
    mat[2, 1] = p32
    mat[2, 2] = p33

    mat[3, 3] = p44

    return mat

cdef dblquad_pcalc_c(DTYPE_t vza):
    cdef np.ndarray[DTYPE_t, ndim=2] mat = np.zeros((4, 4), dtype=np.float)

    mat[0, 0] = -0.16666666666666666 * pi * (-5 + 3 * cos(2 * vza))
    mat[0, 1] = pi * cos(vza) ** 2

    mat[1, 0] = pi / 3
    mat[1, 1] = pi

    mat[3, 3] = pi * cos(vza)

    return mat

cdef quad_pcalc_c(DTYPE_t vza, DTYPE_t raa):
    cdef np.ndarray[DTYPE_t, ndim=2] mat = np.zeros((4, 4), dtype=np.float)

    mat[0, 0] = 0.08333333333333333 * (
                5 + cos(2 * raa) + (-3 + cos(2 * raa)) * cos(2 * vza) + 4 * cos(raa) * sin(2 * vza))
    mat[0, 1] = cos(vza) ** 2 * sin(raa) ** 2
    mat[0, 2] = 0.125 * sin(raa) * (4 * cos(raa) * cos(vza) ** 2 + pi * sin(2 * vza))

    mat[1, 0] = sin(raa) ** 2 / 3
    mat[1, 1] = cos(raa) ** 2
    mat[1, 2] = -0.5 * cos(raa) * sin(raa)

    mat[2, 0] = -0.3333333333333333 * (2 + cos(raa)) * cos(vza) * sin(raa)
    mat[2, 1] = cos(vza) * sin(2 * raa)
    mat[2, 2] = 0.25 * (2 * cos(2 * raa) * cos(vza) + pi * cos(raa) * sin(vza))

    mat[3, 3] = 0.25 * (2 * cos(vza) + pi * cos(raa) * sin(vza))

    return mat

def dblquad_pcalc_c_wrapper(DTYPE_t vza):
    return dblquad_pcalc_c(vza)

def quad_pcalc_c_wrapper(DTYPE_t vza, DTYPE_t raa):
    return quad_pcalc_c(vza, raa)

def dblquad_c_wrapper(DTYPE_t vza, float a, float b, float g, float h):
    return dblquad_c(vza, a, b, g, h)

def quad_c_wrapper(DTYPE_t vza, DTYPE_t raa, float a, float b):
    return quad_c(vza, raa, a, b)

def pmatrix_wrapper(DTYPE_t iza, DTYPE_t vza, DTYPE_t raa):
    return pmatrix_c(iza, vza, raa)
