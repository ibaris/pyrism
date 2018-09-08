from __future__ import division
from ...core.rayleigh_phase import (p11_c_wrapper, p12_c_wrapper, p13_c_wrapper,
                                    p21_c_wrapper,
                                    p22_c_wrapper, p23_c_wrapper, p31_c_wrapper,
                                    p32_c_wrapper,
                                    p33_c_wrapper, p44_c_wrapper)

def p11(iza, raa, vza):
    return p11_c_wrapper(iza, raa, vza, 0)


def p12(iza, raa, vza):
    return p12_c_wrapper(iza, raa, vza, 0)


def p13(iza, raa, vza):
    return p13_c_wrapper(iza, raa, vza, 0)


def p21(iza, raa, vza):
    return p21_c_wrapper(iza, raa, vza, 0)


def p22(iza, raa, vza):
    return p22_c_wrapper(iza, raa, vza, 0)


def p23(iza, raa, vza):
    return p23_c_wrapper(iza, raa, vza, 0)


def p31(iza, raa, vza):
    return p31_c_wrapper(iza, raa, vza, 0)


def p32(iza, raa, vza):
    return p32_c_wrapper(iza, raa, vza, 0)


def p33(iza, raa, vza):
    return p33_c_wrapper(iza, raa, vza, 0)


def p44(iza, raa, vza):
    return p44_c_wrapper(iza, raa, vza, 0)


def VV(iza, raa, vza):
    return p11(iza, raa, vza) + p12(iza, raa, vza) + p13(iza, raa, vza)


def HH(iza, raa, vza):
    return p21(iza, raa, vza) + p22(iza, raa, vza) + p23(iza, raa, vza)


def VH(iza, raa, vza):
    return p31(iza, raa, vza) + p32(iza, raa, vza) + p33(iza, raa, vza)


def HV(iza, raa, vza):
    return p44(iza, raa, vza)
