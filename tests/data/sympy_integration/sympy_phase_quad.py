import numpy as np
import sympy as sym
from sympy import sin, cos, pi

var_iza, var_vza, var_raa = sym.symbols('var_iza var_vza var_raa')


def p11_sym(sym_iza, sym_vza, sym_raa):
    first = sin(sym_vza) ** 2 * sin(sym_iza) ** 2
    second = cos(sym_vza) ** 2 * cos(sym_iza) ** 2 * cos(sym_raa) ** 2
    third = 2 * sin(sym_vza) * sin(sym_iza) * cos(sym_vza) * cos(sym_iza) * cos(sym_raa)

    return first + second + third


def int_p11(sym_iza, sym_vza, sym_raa):
    return sym.simplify(sym.integrate(p11_sym(sym_iza, sym_vza, sym_raa) * sin(sym_iza), (var_iza, 0, sym.pi / 2),
                                      (var_raa, 0, 2 * sym.pi)))


def p12_sym(sym_iza, sym_vza, sym_raa):
    first = cos(sym_vza) ** 2 * sin(sym_raa) ** 2

    return first


def p13_sym(sym_iza, sym_vza, sym_raa):
    first = cos(sym_vza) * sin(sym_vza) * sin(sym_iza) * sin(sym_raa)
    second = cos(sym_vza) ** 2 * cos(sym_iza) * sin(sym_raa) * cos(sym_raa)

    return first + second


def p21_sym(sym_iza, sym_vza, sym_raa):
    first = cos(sym_iza) ** 2 * sin(sym_raa) ** 2

    return first


def p22_sym(sym_iza, sym_vza, sym_raa):
    first = cos(sym_raa) ** 2

    return first


def p23_sym(sym_iza, sym_vza, sym_raa):
    first = -cos(sym_iza) * sin(sym_raa) * cos(sym_raa)

    return first


def p31_sym(sym_iza, sym_vza, sym_raa):
    first = -2 * cos(sym_vza) * sin(sym_iza) * cos(sym_iza) * sin(sym_raa)
    second = - cos(sym_vza) * cos(sym_iza) ** 2 * cos(sym_raa) * sin(sym_raa)

    return first + second


def p32_sym(sym_iza, sym_vza, sym_raa):
    first = 2 * cos(sym_vza) * sin(sym_raa) * cos(sym_raa)

    return first


def p33_sym(sym_iza, sym_vza, sym_raa):
    first = sin(sym_vza) * sin(sym_iza) * cos(sym_raa)
    second = cos(sym_vza) * cos(sym_iza) * cos(2 * sym_raa)

    return first + second


def p44_sym(sym_iza, sym_vza, sym_raa):
    first = sin(sym_vza) * sin(sym_iza) * cos(sym_raa)
    second = cos(sym_vza) * cos(sym_iza)

    return first + second


def phase_matrix_sym(sym_iza, sym_vza, sym_raa):
    mat = np.zeros((4, 4), dtype=np.float)

    mat[0, 0] = p11_sym(sym_iza, sym_vza, sym_raa)
    mat[0, 1] = p12_sym(sym_iza, sym_vza, sym_raa)
    mat[0, 2] = p13_sym(sym_iza, sym_vza, sym_raa)

    mat[1, 0] = p21_sym(sym_iza, sym_vza, sym_raa)
    mat[1, 1] = p22_sym(sym_iza, sym_vza, sym_raa)
    mat[1, 2] = p23_sym(sym_iza, sym_vza, sym_raa)

    mat[2, 0] = p31_sym(sym_iza, sym_vza, sym_raa)
    mat[2, 1] = p32_sym(sym_iza, sym_vza, sym_raa)
    mat[2, 2] = p33_sym(sym_iza, sym_vza, sym_raa)

    mat[3, 3] = p44_sym(sym_iza, sym_vza, sym_raa)

    return mat


def dblquad_evalf_mvza(iza, vza, raa):
    mat = np.zeros((4, 4), dtype=np.float)

    p11 = sym.simplify(
        sym.integrate(p11_sym(var_iza, pi - var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p12 = sym.simplify(
        sym.integrate(p12_sym(var_iza, pi - var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p13 = sym.simplify(
        sym.integrate(p13_sym(var_iza, pi - var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})

    p21 = sym.simplify(
        sym.integrate(p21_sym(var_iza, pi - var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p22 = sym.simplify(
        sym.integrate(p22_sym(var_iza, pi - var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p23 = sym.simplify(
        sym.integrate(p23_sym(var_iza, pi - var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})

    p31 = sym.simplify(
        sym.integrate(p31_sym(var_iza, pi - var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p32 = sym.simplify(
        sym.integrate(p32_sym(var_iza, pi - var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p33 = sym.simplify(
        sym.integrate(p33_sym(var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})

    p44 = sym.simplify(
        sym.integrate(p44_sym(var_iza, pi - var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})

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


def dblquad_evalf_miza(iza, vza, raa):
    mat = np.zeros((4, 4), dtype=np.float)

    p11 = sym.simplify(
        sym.integrate(p11_sym(pi - var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p12 = sym.simplify(
        sym.integrate(p12_sym(pi - var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p13 = sym.simplify(
        sym.integrate(p13_sym(pi - var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})

    p21 = sym.simplify(
        sym.integrate(p21_sym(pi - var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p22 = sym.simplify(
        sym.integrate(p22_sym(pi - var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p23 = sym.simplify(
        sym.integrate(p23_sym(pi - var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})

    p31 = sym.simplify(
        sym.integrate(p31_sym(pi - var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p32 = sym.simplify(
        sym.integrate(p32_sym(pi - var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p33 = sym.simplify(
        sym.integrate(p33_sym(pi - var_iza, vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})

    p44 = sym.simplify(
        sym.integrate(p44_sym(pi - var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})

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


def dblquad_evalf(iza, vza, raa):
    mat = np.zeros((4, 4), dtype=np.float)

    p11 = sym.simplify(
        sym.integrate(p11_sym(var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p12 = sym.simplify(
        sym.integrate(p12_sym(var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p13 = sym.simplify(
        sym.integrate(p13_sym(var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})

    p21 = sym.simplify(
        sym.integrate(p21_sym(var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p22 = sym.simplify(
        sym.integrate(p22_sym(var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p23 = sym.simplify(
        sym.integrate(p23_sym(var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})

    p31 = sym.simplify(
        sym.integrate(p31_sym(var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p32 = sym.simplify(
        sym.integrate(p32_sym(var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})
    p33 = sym.simplify(
        sym.integrate(p33_sym(var_iza, vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})

    p44 = sym.simplify(
        sym.integrate(p44_sym(var_iza, var_vza, var_raa) * sin(var_iza), (var_iza, 0, sym.pi / 2),
                      (var_raa, 0, 2 * sym.pi))).evalf(subs={var_iza: iza, var_vza: vza, var_raa: raa})

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


def dblquad(iza, vza, raa):
    mat = np.zeros((4, 4), dtype=np.object)

    p11 = sym.simplify(
        sym.integrate(p11_sym(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2),
                      (raa, 0, 2 * sym.pi)))
    p12 = sym.simplify(
        sym.integrate(p12_sym(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2),
                      (raa, 0, 2 * sym.pi)))
    p13 = sym.simplify(
        sym.integrate(p13_sym(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2),
                      (raa, 0, 2 * sym.pi)))

    p21 = sym.simplify(
        sym.integrate(p21_sym(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2),
                      (raa, 0, 2 * sym.pi)))
    p22 = sym.simplify(
        sym.integrate(p22_sym(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2),
                      (raa, 0, 2 * sym.pi)))
    p23 = sym.simplify(
        sym.integrate(p23_sym(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2),
                      (raa, 0, 2 * sym.pi)))

    p31 = sym.simplify(
        sym.integrate(p31_sym(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2),
                      (raa, 0, 2 * sym.pi)))
    p32 = sym.simplify(
        sym.integrate(p32_sym(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2),
                      (raa, 0, 2 * sym.pi)))
    p33 = sym.simplify(
        sym.integrate(p33_sym(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2),
                      (raa, 0, 2 * sym.pi)))

    p44 = sym.simplify(
        sym.integrate(p44_sym(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2),
                      (raa, 0, 2 * sym.pi)))

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
