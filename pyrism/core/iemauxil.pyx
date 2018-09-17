# cython: cdivision=True
from __future__ import division

cimport numpy as np
import numpy as np
from libc.math cimport cos, sin, sqrt, exp, abs
import cmath
from scipy.integrate import dblquad
import cython
from scipy.special import erf

# cimport libcpp.double complex                     # imports classes and functions
# from libcpp.double complex cimport *          # imports the operators

DTYPE = np.float

ctypedef np.float_t DTYPE_t

# cdef extern from "complex.h":
#     pass

def factorial(int x):
    # Basic example of a cython function, which defines
    # python-like operations and control flow on defined c types

    cdef int m = x
    cdef int i

    if x <= 1:
        return 1
    else:
        for i in range(1, x):
            m = m * i
        return m

def reflection_coefficients(float k, double complex iza, double complex vza, double complex raa, double complex phi,
                            double complex eps,
                            double complex sigma):
    cdef DTYPE_t error

    cdef double complex rt = np.sqrt(eps - pow(np.sin(iza + 0.01), 2))
    cdef double complex Rvi = (eps * np.cos(iza + 0.01) - rt) / (eps * np.cos(iza + 0.01) + rt)
    cdef double complex Rhi = (np.cos(iza + 0.01) - rt) / (np.cos(iza + 0.01) + rt)
    cdef double complex wvnb = k * np.sqrt(
        pow((np.sin(vza) * np.cos(raa) - np.sin(iza + 0.01) * np.cos(phi)), 2) + pow((
                np.sin(vza) * np.sin(raa) - np.sin(iza + 0.01) * np.sin(phi)), 2))

    cdef int Ts = 1

    cdef float merror = 1.0e8
    while merror >= 1.0e-3 and Ts <= 150:
        Ts += 1
        error = ((k * sigma) ** 2 * (np.cos(iza + 0.01) + np.cos(vza)) ** 2) ** Ts / factorial(Ts)
        merror = np.mean(error)

    return rt, Rvi, Rhi, wvnb, Ts

def r_transition(float k, double complex iza, double complex vza, double complex sigma, double complex eps,
                 np.ndarray Wn, int Ts):
    cdef int i
    cdef double complex a0

    cdef double complex Rv0 = (cmath.sqrt(eps) - 1) / (cmath.sqrt(eps) + 1)
    cdef double complex Rh0 = -Rv0

    cdef double complex Ft = 8 * Rv0 ** 2 * cmath.sin(vza) * (
            np.cos(iza + 0.01) + cmath.sqrt(eps - cmath.sin(iza + 0.01) ** 2)) / (
                                     np.cos(iza + 0.01) * cmath.sqrt(eps - cmath.sin(iza + 0.01) ** 2))
    cdef double complex a1 = 0.0
    cdef double complex b1 = 0.0

    for i in range(Ts):
        i += 1
        a0 = ((k * sigma) * np.cos(iza + 0.01)) ** (2 * i) / factorial(i)
        a1 = a1 + a0 * Wn[i - 1]
        b1 = b1 + a0 * (
            abs(Ft / 2 + 2 ** (i + 1) * Rv0 / np.cos(iza + 0.01) * np.exp(
                - ((k * sigma) * np.cos(iza + 0.01)) ** 2))) ** 2 * Wn[
                 i - 1]

    cdef double complex St = 0.25 * (abs(Ft) ** 2) * a1 / b1
    cdef double complex St0 = 1 / (abs(1 + 8 * Rv0 / (np.cos(iza + 0.01) * Ft))) ** 2
    cdef double complex Tf = 1 - St / St0

    return Tf, Rv0, Rh0

def RaV_integration_function(DTYPE_t Zy, DTYPE_t Zx, double complex iza, double complex sigx, double complex sigy,
                             double complex eps):
    cdef double complex A = np.cos(iza + 0.01) + Zx * cmath.sin(iza + 0.01)
    cdef double complex B = (1 + Zx ** 2 + Zy ** 2) * eps

    cdef double complex CC = cmath.sin(iza + 0.01) ** 2 - 2 * Zx * cmath.sin(iza + 0.01) * np.cos(
        iza + 0.01) + Zx ** 2 * np.cos(
        iza + 0.01) ** 2 + Zy ** 2

    cdef double complex Rv = (eps * A - np.sqrt(B - CC)) / (eps * A + np.sqrt(B - CC))

    cdef double complex pd = np.exp(-Zx ** 2 / (2 * sigx ** 2) - Zy ** 2 / (2 * sigy ** 2))

    cdef double complex Rav = Rv * pd

    return Rav.real

def RaH_integration_function(DTYPE_t Zy, DTYPE_t Zx, double complex iza, double complex sigx, double complex sigy,
                             double complex eps):
    cdef double complex A = np.cos(iza + 0.01) + Zx * cmath.sin(iza + 0.01)
    cdef double complex B = eps * (1 + Zx ** 2 + Zy ** 2)

    cdef double complex CC = cmath.sin(iza + 0.01) ** 2 - 2 * Zx * cmath.sin(iza + 0.01) * np.cos(
        iza + 0.01) + Zx ** 2 * np.cos(
        iza + 0.01) ** 2 + Zy ** 2

    cdef double complex Rh = (A - np.sqrt(B - CC)) / (A + np.sqrt(B - CC))

    cdef double complex pd = np.exp(-Zx ** 2 / (2 * sigx ** 2) - Zy ** 2.0 / (2 * sigy ** 2))

    cdef double complex RaH = Rh * pd

    return RaH.real

def RaV_integration(double complex iza, double complex sigx, double complex sigy, double complex eps):
    cdef double complex ravv, Rav

    cdef float bound = 3 * sigx.real

    ravv = dblquad(RaV_integration_function, -bound, bound, lambda x: -bound, lambda x: bound,
                   args=(iza, sigx, sigy, eps))[0]
    temp = np.asarray(ravv) / (2 * np.pi * sigx * sigy)

    Rav = np.asarray(np.asarray(ravv) / (2 * np.pi * sigx * sigy))

    return Rav

def RaH_integration(double complex iza, double complex sigx, double complex sigy, double complex eps):
    cdef double complex rahh, RaH

    cdef float bound = 3 * sigx.real

    rahh = dblquad(RaH_integration_function, -bound, bound, lambda x: -bound, lambda x: bound,
                   args=(iza, sigx, sigy, eps))[0]

    RaH = np.asarray(np.asarray(rahh) / (2 * np.pi * sigx * sigy))

    return RaH

def Ra_integration(double complex iza, double complex sigma, double complex corrlength, double complex eps):
    cdef double complex sigx = 1.1 * sigma / corrlength
    cdef double complex sigy = sigx

    cdef double complex RaV = RaV_integration(iza, sigx, sigy, eps)
    cdef double complex RaH = RaH_integration(iza, sigx, sigy, eps)

    return RaV, RaH

def biStatic_coefficient(DTYPE_t iza, DTYPE_t vza, DTYPE_t raa, DTYPE_t Rvi, DTYPE_t Rv0, DTYPE_t Rhi, DTYPE_t Rh0,
                         DTYPE_t RaV, DTYPE_t RaH, DTYPE_t Tf):
    cdef DTYPE_t Rvt, Rht, fvv, fhh

    if np.array_equal(vza, iza) and (np.allclose(np.all(raa), np.pi)):
        Rvt = Rvi + (Rv0 - Rvi) * Tf
        Rht = Rhi + (Rh0 - Rhi) * Tf

    else:
        Rvt = RaV
        Rht = RaH

    fvv = 2 * Rvt * (
            sin(iza + 0.01) * sin(vza) - (1 + np.cos(iza + 0.01) * np.cos(vza)) * np.cos(raa)) / (
                  np.cos(iza + 0.01) + np.cos(vza))

    fhh = -2 * Rht * (
            sin(iza + 0.01) * sin(vza) - (1 + np.cos(iza + 0.01) * np.cos(vza)) * np.cos(raa)) / (
                  np.cos(iza + 0.01) + np.cos(vza))

    return fvv, fhh, Rvt, Rht

def Fppupdn_calc(double complex ud, int method, double complex Rvi, double complex Rhi, double complex er,
                 float k, float kz, float ksz, double complex s, double complex cs, ss, css,
                 double complex cf, double complex cfs, double complex sfs):
    cdef double complex Gqi, Gqti, qi, c11, c21, c31, c41, c51, c12, c22, c32, c42, c52, q, qt, vv, hh
    if method == 1:
        Gqi = ud * kz
        Gqti = ud * k * np.sqrt(er - s ** 2)
        qi = ud * kz

        c11 = k * cfs * (ksz - qi)
        c21 = cs * (cfs * (
                k ** 2 * s * cf * (ss * cfs - s * cf) + Gqi * (k * css - qi)) + k ** 2 * cf * s * ss * sfs ** 2)
        c31 = k * s * (s * cf * cfs * (k * css - qi) - Gqi * (cfs * (ss * cfs - s * cf) + ss * sfs ** 2))
        c41 = k * cs * (cfs * css * (k * css - qi) + k * ss * (ss * cfs - s * cf))
        c51 = Gqi * (cfs * css * (qi - k * css) - k * ss * (ss * cfs - s * cf))

        c12 = k * cfs * (ksz - qi)
        c22 = cs * (cfs * (
                k ** 2 * s * cf * (ss * cfs - s * cf) + Gqti * (k * css - qi)) + k ** 2 * cf * s * ss * sfs ** 2)
        c32 = k * s * (s * cf * cfs * (k * css - qi) - Gqti * (cfs * (ss * cfs - s * cf) - ss * sfs ** 2))
        c42 = k * cs * (cfs * css * (k * css - qi) + k * ss * (ss * cfs - s * cf))
        c52 = Gqti * (cfs * css * (qi - k * css) - k * ss * (ss * cfs - s * cf))

    if method == 2:
        Gqs = ud * ksz
        Gqts = ud * k * np.sqrt(er - ss ** 2)
        qs = ud * ksz

        c11 = k * cfs * (kz + qs)
        c21 = Gqs * (cfs * (cs * (k * cs + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs ** 2)
        c31 = k * ss * (k * cs * (ss * cfs - s * cf) + s * (kz + qs))
        c41 = k * css * (cfs * (cs * (kz + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs ** 2)
        c51 = -css * (k ** 2 * ss * (ss * cfs - s * cf) + Gqs * cfs * (kz + qs))

        c12 = k * cfs * (kz + qs)
        c22 = Gqts * (cfs * (cs * (kz + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs ** 2)
        c32 = k * ss * (k * cs * (ss * cfs - s * cf) + s * (kz + qs))
        c42 = k * css * (cfs * (cs * (kz + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs ** 2)
        c52 = -css * (k ** 2 * ss * (ss * cfs - s * cf) + Gqts * cfs * (kz + qs))

    q = kz
    qt = k * np.sqrt(er - s ** 2)

    vv = (1 + Rvi) * (-(1 - Rvi) * c11 / q + (1 + Rvi) * c12 / qt) + \
         (1 - Rvi) * ((1 - Rvi) * c21 / q - (1 + Rvi) * c22 / qt) + \
         (1 + Rvi) * ((1 - Rvi) * c31 / q - (1 + Rvi) * c32 / er / qt) + \
         (1 - Rvi) * ((1 + Rvi) * c41 / q - er * (1 - Rvi) * c42 / qt) + \
         (1 + Rvi) * ((1 + Rvi) * c51 / q - (1 - Rvi) * c52 / qt)

    hh = (1 + Rhi) * ((1 - Rhi) * c11 / q - er * (1 + Rhi) * c12 / qt) - \
         (1 - Rhi) * ((1 - Rhi) * c21 / q - (1 + Rhi) * c22 / qt) - \
         (1 + Rhi) * ((1 - Rhi) * c31 / q - (1 + Rhi) * c32 / qt) - \
         (1 - Rhi) * ((1 + Rhi) * c41 / q - (1 - Rhi) * c42 / qt) - \
         (1 + Rhi) * ((1 + Rhi) * c51 / q - (1 - Rhi) * c52 / qt)

    return vv, hh

def Ipp(double complex siza, double complex ciza, double complex svza, double complex cvza,
        double complex sraa, double complex craa, double complex cphi, double complex Rvi,
        double complex Rhi, double complex eps, float k, float kz_iza, float kz_vza,
        double complex fvv, double complex fhh, double complex sigma, int Ts):
    cdef int i
    Fvvupi, Fhhupi = Fppupdn_calc(+1, 1,
                                  Rvi,
                                  Rhi,
                                  eps,
                                  k,
                                  kz_iza,
                                  kz_vza,
                                  siza,
                                  ciza,
                                  svza,
                                  cvza,
                                  cphi,
                                  craa,
                                  sraa)

    Fvvups, Fhhups = Fppupdn_calc(+1, 2,
                                  Rvi,
                                  Rhi,
                                  eps,
                                  k,
                                  kz_iza,
                                  kz_vza,
                                  siza,
                                  ciza,
                                  svza,
                                  cvza,
                                  cphi,
                                  craa,
                                  sraa)

    Fvvdni, Fhhdni = Fppupdn_calc(-1, 1,
                                  Rvi,
                                  Rhi,
                                  eps,
                                  k,
                                  kz_iza,
                                  kz_vza,
                                  siza,
                                  ciza,
                                  svza,
                                  cvza,
                                  cphi,
                                  craa,
                                  sraa)

    Fvvdns, Fhhdns = Fppupdn_calc(-1, 2,
                                  Rvi,
                                  Rhi,
                                  eps,
                                  k,
                                  kz_iza,
                                  kz_vza,
                                  siza,
                                  ciza,
                                  svza,
                                  cvza,
                                  cphi,
                                  craa,
                                  sraa)

    cdef double complex qi = k * ciza
    cdef double complex qs = k * cvza

    ivv = []
    ihh = []
    for i in range(Ts):
        i += 1
        Ivv = (kz_iza + kz_vza) ** i * fvv * np.exp(
            -sigma ** 2 * kz_iza * kz_vza) + \
              0.25 * (Fvvupi * (kz_vza - qi) ** (i - 1) * np.exp(
            -sigma ** 2 * (qi ** 2 - qi * (kz_vza - kz_iza))) + Fvvdni * (
                              kz_vza + qi) ** (i - 1) * np.exp(
            -sigma ** 2 * (qi ** 2 + qi * (kz_vza - kz_iza))) + Fvvups * (
                              kz_iza + qs) ** (i - 1) * np.exp(
            -sigma ** 2 * (qs ** 2 - qs * (kz_vza - kz_iza))) + Fvvdns * (
                              kz_iza - qs) ** (i - 1) * np.exp(
            -sigma ** 2 * (qs ** 2 + qs * (kz_vza - kz_iza))))

        Ihh = (kz_iza + kz_vza) ** i * fhh * np.exp(
            -sigma ** 2 * kz_iza * kz_vza) + \
              0.25 * (Fhhupi * (kz_vza - qi) ** (i - 1) * np.exp(
            -sigma ** 2 * (qi ** 2 - qi * (kz_vza - kz_iza))) +
                      Fhhdni * (kz_vza + qi) ** (i - 1) * np.exp(
                    -sigma ** 2 * (qi ** 2 + qi * (kz_vza - kz_iza))) +
                      Fhhups * (kz_iza + qs) ** (i - 1) * np.exp(
                    -sigma ** 2 * (qs ** 2 - qs * (kz_vza - kz_iza))) +
                      Fhhdns * (kz_iza - qs) ** (i - 1) * np.exp(
                    -sigma ** 2 * (qs ** 2 + qs * (kz_vza - kz_iza))))

        ivv.append(Ivv)
        ihh.append(Ihh)
    Ivv = ivv
    Ihh = ihh

    return Ivv, Ihh

def sigma_nought(int Ts, np.ndarray Wn, np.ndarray Ivv, np.ndarray Ihh, double complex ShdwS, float k,
                 float kz_iza, float kz_vza, double complex sigma):
    cdef float sigmavv = 0
    cdef float sigmahh = 0
    for i in range(Ts):
        i += 1
        a0 = Wn[i - 1] / factorial(i) * sigma ** (2 * i)

        sigmavv = sigmavv + np.abs(Ivv[i - 1]) ** 2 * a0
        sigmahh = sigmahh + np.abs(Ihh[i - 1]) ** 2 * a0

    cdef float VV = sigmavv * ShdwS * k ** 2 / 2 * np.exp(
        -sigma ** 2 * (kz_iza ** 2 + kz_vza ** 2))
    cdef float HH = sigmahh * ShdwS * k ** 2 / 2 * np.exp(
        -sigma ** 2 * (kz_iza ** 2 + kz_vza ** 2))

    return VV, HH

def shadowing_function(iza, vza, raa, rss):
    if np.array_equal(vza, iza) and (np.allclose(np.all(raa), np.pi)):
        ct = np.cos(iza) / np.sin(iza)
        cts = np.cos(vza) / np.sin(vza)
        rslp = rss
        ctorslp = ct / np.sqrt(2) / rslp
        ctsorslp = cts / np.sqrt(2) / rslp
        shadf = 0.5 * (np.exp(-ctorslp ** 2) / np.sqrt(np.pi) / ctorslp - erf(ctorslp))
        shadfs = 0.5 * (np.exp(-ctsorslp ** 2) / np.sqrt(np.pi) / ctsorslp - erf(ctsorslp))
        ShdwS = 1 / (1 + shadf + shadfs)
    else:
        ShdwS = 1

    return ShdwS

def calc_i2em_auxil(float k, float kz_iza, float kz_vza, double complex iza, double complex vza, double complex raa,
                    double complex phi, double complex eps, double complex corrlength, double complex sigma,
                    corrfunc, int n):
    cdef double complex rt, Rvi, Rhi, wvnb, Tf, Rv0, Rh0, RaV, RaH
    cdef int Ts
    cdef list Ivv, Ihh
    cdef np.ndarray Ivva, Ihha

    cdef double complex siza = np.sin(iza + 0.01)
    cdef double complex ciza = np.cos(iza + 0.01)
    cdef double complex svza = np.sin(vza)
    cdef double complex cvza = np.cos(vza)
    cdef double complex sraa = np.sin(raa)
    cdef double complex craa = np.cos(raa)
    cdef double complex cphi = np.cos(phi)

    rt, Rvi, Rhi, wvnb, Ts = reflection_coefficients(k, iza, vza, raa, phi, eps, sigma)

    Wn, rss = corrfunc(sigma.real, corrlength.real, wvnb.real, Ts, n=n)
    ShdwS = shadowing_function(iza.real, vza.real, raa.real, rss)

    Tf, Rv0, Rh0 = r_transition(k, iza, vza, sigma, eps, Wn, Ts)
    RaV, RaH = Ra_integration(iza, sigma, corrlength, eps)
    fvv, fhh, Rvt, Rht = biStatic_coefficient(iza.real, vza.real, raa.real, Rvi.real, Rv0.real, Rhi.real, Rh0.real,
                                              RaV.real, RaH.real,
                                              Tf.real)

    Ivv, Ihh = Ipp(siza, ciza, svza, cvza, sraa, craa, cphi, Rvi, Rhi, eps, k,
                   kz_iza, kz_vza, fvv, fhh, sigma, Ts)

    Ivva, Ihha = np.asarray(Ivv, dtype=np.complex), np.asarray(Ihh, dtype=np.complex)

    VV, HH = sigma_nought(Ts, Wn, Ivva, Ihha, ShdwS, k, kz_iza, kz_vza, sigma.real)

    return VV, HH

def emsv_integralfunc(float x, float y, float iza, double complex eps, double complex rv, double complex rh, float k,
                      float kl, float ks,
                      double complex sq, corrfunc, float corrlength, int pol):
    cdef int nr
    cdef np.ndarray wn, expwn, gauswn, svv, shh
    cdef float wnn, vv, hv, ref, hh, vh
    cdef double complex Fvv, Fhv, Fvvs, Fhvs, Ivv, Ihv, Ihh, Ivh

    cdef float error = 1.0e3
    cdef double complex sqs = np.sqrt(eps - sin(x) ** 2)
    cdef double complex rc = (rv - rh) / 2
    cdef double complex tv = 1 + rv
    cdef double complex th = 1 + rh

    # -- calc coefficients for surface correlation spectra
    cdef float wvnb = k * np.sqrt(sin(iza) ** 2 - 2 * sin(iza) * sin(x) * cos(y) + sin(x) ** 2)

    try:
        nr = len(x)
    except (IndexError, TypeError):
        nr = 1

    # -- calculate number of spectral components needed
    cdef int n_spec = 1
    while error > 1.0e-3:
        n_spec = n_spec + 1
        #   error = (ks2 *(cs + css)**2 )**n_spec / factorial(n_spec)
        # ---- in this case we will use the smallest ths to determine the number of
        # spectral components to use.  It might be more than needed for other angles
        # but this is fine.  This option is used to simplify calculations.
        error = (ks ** 2 * (cos(iza) + cos(x)) ** 2) ** n_spec / factorial(n_spec)
        error = np.min(error)
    # -- calculate expressions for the surface spectra

    if corrfunc == 1:
        wn = np.zeros([n_spec, nr])

        for n in range(n_spec):
            wn[n, :] = (n + 1) * kl ** 2 / ((n + 1) ** 2 + (wvnb * corrlength) ** 2) ** 1.5

    if corrfunc == 2:
        wn = np.zeros([n_spec, nr])

        for n in range(n_spec):
            wn[n, :] = 0.5 * kl ** 2 / (n + 1) * np.exp(-(wvnb * corrlength) ** 2 / (4 * (n + 1)))

    if corrfunc == 3:
        expwn = np.zeros([n_spec, nr])
        gauswn = np.zeros([n_spec, nr])

        for n in range(n_spec):
            expwn[n, :] = (n + 1) * kl ** 2 / ((n + 1) ** 2 + (wvnb * corrlength) ** 2) ** 1.5
            gauswn[n, :] = 0.5 * kl ** 2 / (n + 1) * np.exp(-(wvnb * corrlength) ** 2 / (4 * (n + 1)))

        wn = expwn / gauswn

    # -- calculate fpq!

    cdef float ff = 2 * (sin(iza) * sin(x) - (1 + cos(iza) * cos(x)) * cos(y)) / (
            cos(iza) + cos(x))

    cdef double complex fvv = rv * ff
    cdef double complex fhh = -rh * ff

    cdef double complex fvh = -2 * rc * sin(y)
    # cdef double complex fhv = 2 * rc * sin(y)

    # -- calculate Fpq and Fpqs -----
    cdef double complex fhv = sin(iza) * (sin(x) - sin(iza) * cos(y)) / (cos(iza) ** 2 * cos(x))
    cdef double complex T = (sq * (cos(iza) + sq) + cos(iza) * (eps * cos(iza) + sq)) / (
            eps * cos(iza) * (cos(iza) + sq) + sq * (eps * cos(iza) + sq))

    cdef double complex cm2 = cos(x) * sq / cos(iza) / sqs - 1
    cdef float ex = np.exp(-ks ** 2 * cos(iza) * cos(x))
    cdef float de = 0.5 * np.exp(-ks ** 2 * (cos(iza) ** 2 + cos(x) ** 2))

    if pol == 1:
        Fvv = (eps - 1) * sin(iza) ** 2 * tv ** 2 * fhv / eps ** 2
        Fhv = (T * sin(iza) * sin(iza) - 1. + cos(iza) / cos(x) + (
                eps * T * cos(iza) * cos(x) * (
                eps * T - sin(iza) * sin(iza)) - sq * sq) / (
                       T * eps * sq * cos(x))) * (1 - rc * rc) * sin(y)

        Fvvs = -cm2 * sq * tv ** 2 * (
                cos(y) - sin(iza) * sin(x)) / cos(
            iza) ** 2 / eps - cm2 * sqs * tv ** 2 * cos(y) / eps - (
                       cos(x) * sq / cos(iza) / sqs / eps - 1) * sin(
            x) * tv ** 2 * (
                       sin(iza) - sin(x) * cos(y)) / cos(iza)
        Fhvs = -(sin(x) * sin(x) / T - 1 + cos(x) / cos(iza) + (
                cos(iza) * cos(x) * (
                1 - sin(x) * sin(x) * T) - T * T * sqs * sqs) / (
                         T * sqs * cos(iza))) * (1 - rc * rc) * sin(y)

        # -- calculate the bistatic field coefficients ---

        svv = np.zeros([n_spec, nr])
        for n in range(n_spec):
            Ivv = fvv * ex * (ks * (cos(iza) + cos(x))) ** (n + 1) + (
                    Fvv * (ks * cos(x)) ** (n + 1) + Fvvs * (ks * cos(iza)) ** (n + 1)) / 2
            Ihv = fhv * ex * (ks * (cos(iza) + cos(x))) ** (n + 1) + (
                    Fhv * (ks * cos(x)) ** (n + 1) + Fhvs * (ks * cos(iza)) ** (n + 1)) / 2

        wnn = wn[n, :] / factorial(n + 1)
        vv = wnn * (abs(Ivv)) ** 2
        hv = wnn * (abs(Ihv)) ** 2
        svv[n, :] = (de * (vv + hv) * sin(x) * (1 / cos(iza))) / (4 * np.pi)

        ref = np.sum([svv])  # adding all n terms stores in different rows

    if pol == 2:
        Fhh = -(eps - 1) * th ** 2 * fhv
        Fvh = (sin(iza) * sin(iza) / T - 1. + cos(iza) / cos(x) + (
                cos(iza) * cos(x) * (
                1 - sin(iza) * sin(iza) * T) - T * T * sq * sq) / (
                       T * sq * cos(x))) * (1 - rc * rc) * sin(y)

        Fhhs = cm2 * sq * th ** 2 * (
                cos(y) - sin(iza) * sin(x)) / cos(
            iza) ** 2 + cm2 * sqs * th ** 2 * cos(y) + cm2 * sin(x) * th ** 2 * (
                       sin(iza) - sin(x) * cos(y)) / cos(iza)
        Fvhs = -(T * sin(x) * sin(x) - 1 + cos(x) / cos(iza) + (
                eps * T * cos(iza) * cos(x) * (
                eps * T - sin(x) * sin(x)) - sqs * sqs) / (
                         T * eps * sqs * cos(iza))) * (1 - rc * rc) * sin(y)

        shh = np.zeros([n_spec, nr])
        for n in range(n_spec):
            Ihh = fhh * ex * (ks * (cos(iza) + cos(x))) ** (n + 1) + (
                    Fhh * (ks * cos(x)) ** (n + 1) + Fhhs * (ks * cos(iza)) ** (n + 1)) / 2
            Ivh = fvh * ex * (ks * (cos(iza) + cos(x))) ** (n + 1) + (
                    Fvh * (ks * cos(x)) ** (n + 1) + Fvhs * (ks * cos(iza)) ** (n + 1)) / 2

        wnn = wn[n, :] / factorial(n + 1)
        hh = wnn * (abs(Ihh)) ** 2
        vh = wnn * (abs(Ivh)) ** 2
        (2 * (3 + 4) * sin(5) * 1 / cos(6)) / (np.pi * 4)
        shh[n, :] = (de * (hh + vh) * sin(x) * (1 / cos(iza))) / (4 * np.pi)

        ref = np.sum([shh])

    return ref

cdef calc_iem_ems(float iza, float k, float sigma, float corrlength, double complex eps, int corrfunc_ems):
    cdef float ks = k * sigma
    cdef float kl = k * corrlength

    # -- calculation of reflection coefficients
    cdef double complex sq = np.sqrt(eps - np.sin(iza) ** 2)

    cdef double complex rv = (eps * np.cos(iza) - sq) / (
            eps * np.cos(iza) + sq)

    cdef double complex rh = (np.cos(iza) - sq) / (np.cos(iza) + sq)

    cdef float refv = dblquad(emsv_integralfunc, 0, np.pi / 2, lambda x: 0, lambda x: np.pi,
                              args=(iza, eps, rv, rh, k, kl, ks, sq, corrfunc_ems, corrlength, 1))[0]

    cdef float refh = dblquad(emsv_integralfunc, 0, np.pi / 2, lambda x: 0, lambda x: np.pi,
                              args=(
                                  iza, eps, rv, rh, k, kl, ks, sq, corrfunc_ems, corrlength,
                                  2))[0]

    cdef float VV = 1 - refv - np.exp(-ks ** 2 * np.cos(iza) * np.cos(iza)) * (
        abs(rv)) ** 2
    cdef float HH = 1 - refh - np.exp(-ks ** 2 * np.cos(iza) * np.cos(iza)) * (
        abs(rh)) ** 2

    return VV, HH

def calc_iem_ems_wrapper(float iza, float k, float sigma, float corrlength, double complex eps, int corrfunc_ems):
    return calc_iem_ems(iza, k, sigma, corrlength, eps, corrfunc_ems)
