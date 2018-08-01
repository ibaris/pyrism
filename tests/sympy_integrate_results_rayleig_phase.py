import numpy as np
import sympy as sym
from sympy import sin, cos

iza, vza, raa = sym.symbols('iza vza raa')


def p11_c(iza, vza, raa):
    first = sin(vza) ** 2 * sin(iza) ** 2
    second = cos(vza) ** 2 * cos(iza) ** 2 * cos(raa) ** 2
    third = 2 * sin(vza) * sin(iza) * cos(vza) * cos(iza) * cos(raa)

    return first + second + third


def p12_c(iza, vza, raa):
    first = cos(vza) ** 2 * sin(raa) ** 2

    return first


def p13_c(iza, vza, raa):
    first = cos(vza) * sin(vza) * sin(iza) * sin(raa)
    second = cos(vza) ** 2 * cos(iza) * sin(raa) * cos(raa)

    return first + second


def p21_c(iza, vza, raa):
    first = cos(iza) ** 2 * sin(raa) ** 2

    return first


def p22_c(iza, vza, raa):
    first = cos(raa) ** 2

    return first


def p23_c(iza, vza, raa):
    first = -cos(iza) * sin(raa) * cos(raa)

    return first


def p31_c(iza, vza, raa):
    first = -2 * cos(vza) * sin(iza) * cos(iza) * sin(raa)
    second = - cos(vza) * cos(iza) ** 2 * cos(raa) * sin(raa)

    return first + second


def p32_c(iza, vza, raa):
    first = 2 * cos(vza) * sin(raa) * cos(raa)

    return first


def p33_c(iza, vza, raa):
    first = sin(vza) * sin(iza) * cos(raa)
    second = cos(vza) * cos(iza) * cos(2 * raa)

    return first + second


def p44_c(iza, vza, raa):
    first = sin(vza) * sin(iza) * cos(raa)
    second = cos(vza) * cos(iza)

    return first + second


def phase_matrix_c(iza, vza, raa):
    mat = np.zeros((4, 4), dtype=np.float)

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


# After that we specify the sensing geometry we want to simulate
iza_ = np.radians(35)  # Incidence zenith angle
vza_ = np.radians(30)  # Viewing zenith angle
raa_ = np.radians(50)  # Relative azimuth angle

p11 = sym.simplify(sym.integrate(p11_c(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2), (raa, 0, 2 * sym.pi))).evalf(
    subs={iza: iza_, vza: vza_, raa: raa_})
p12 = sym.simplify(sym.integrate(p12_c(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2), (raa, 0, 2 * sym.pi))).evalf(
    subs={iza: iza_, vza: vza_, raa: raa_})
p13 = sym.simplify(sym.integrate(p13_c(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2), (raa, 0, 2 * sym.pi))).evalf(
    subs={iza: iza_, vza: vza_, raa: raa_})

p21 = sym.simplify(sym.integrate(p21_c(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2), (raa, 0, 2 * sym.pi))).evalf(
    subs={iza: iza_, vza: vza_, raa: raa_})
p22 = sym.simplify(sym.integrate(p22_c(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2), (raa, 0, 2 * sym.pi))).evalf(
    subs={iza: iza_, vza: vza_, raa: raa_})
p23 = sym.simplify(sym.integrate(p23_c(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2), (raa, 0, 2 * sym.pi))).evalf(
    subs={iza: iza_, vza: vza_, raa: raa_})

p31 = sym.simplify(sym.integrate(p31_c(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2), (raa, 0, 2 * sym.pi))).evalf(
    subs={iza: iza_, vza: vza_, raa: raa_})
p32 = sym.simplify(sym.integrate(p32_c(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2), (raa, 0, 2 * sym.pi))).evalf(
    subs={iza: iza_, vza: vza_, raa: raa_})
p33 = sym.simplify(sym.integrate(p33_c(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2), (raa, 0, 2 * sym.pi))).evalf(
    subs={iza: iza_, vza: vza_, raa: raa_})

p44 = sym.simplify(sym.integrate(p44_c(iza, vza, raa) * sin(iza), (iza, 0, sym.pi / 2), (raa, 0, 2 * sym.pi))).evalf(
    subs={iza: iza_, vza: vza_, raa: raa_})

mat = np.zeros((4, 4), dtype=np.float)

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
