# -*- coding: utf-8 -*-
from __future__ import division

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.integrate import dblquad
from scipy.misc import factorial
from scipy.special import erf

from ....core import (cot)

# python 3.6 comparability
if sys.version_info < (3, 0):
    srange = xrange
else:
    srange = range


def reflection_coefficients(k, iza, vza, raa, phi, eps, sigma):
    rt = np.sqrt(eps - np.sin(iza + 0.01) ** 2)
    Rvi = (eps * np.cos(iza + 0.01) - rt) / (eps * np.cos(iza + 0.01) + rt)
    Rhi = (np.cos(iza + 0.01) - rt) / (np.cos(iza + 0.01) + rt)
    wvnb = k * np.sqrt(
        (np.sin(vza) * np.cos(raa) - np.sin(iza + 0.01) * np.cos(phi)) ** 2 + (
                np.sin(vza) * np.sin(raa) - np.sin(iza + 0.01) * np.sin(phi)) ** 2)
    Ts = 1

    merror = 1.0e8
    while merror >= 1.0e-3 and Ts <= 150:
        Ts += 1
        error = ((k * sigma) ** 2 * (
                np.cos(iza + 0.01) + np.cos(vza)) ** 2) ** Ts / factorial(Ts)
        merror = error.mean()

    return rt, Rvi, Rhi, wvnb, Ts


def r_transition(k, iza, vza, sigma, eps, Wn, Ts):
    Rv0 = (np.sqrt(eps) - 1) / (np.sqrt(eps) + 1)
    Rh0 = -Rv0

    Ft = 8 * Rv0 ** 2 * np.sin(vza) * (
            np.cos(iza + 0.01) + np.sqrt(eps - np.sin(iza + 0.01) ** 2)) / (
                 np.cos(iza + 0.01) * np.sqrt(eps - np.sin(iza + 0.01) ** 2))
    a1 = 0
    b1 = 0

    for i in srange(Ts):
        i += 1
        a0 = ((k * sigma) * np.cos(iza + 0.01)) ** (2 * i) / factorial(i)
        a1 = a1 + a0 * Wn[i - 1]
        b1 = b1 + a0 * (np.abs(
            Ft / 2 + 2 ** (i + 1) * Rv0 / np.cos(iza + 0.01) * np.exp(
                - ((k * sigma) * np.cos(iza + 0.01)) ** 2))) ** 2 * Wn[i - 1]

    St = 0.25 * (np.abs(Ft) ** 2) * a1 / b1
    St0 = 1 / (np.abs(1 + 8 * Rv0 / (np.cos(iza + 0.01) * Ft))) ** 2
    Tf = 1 - St / St0

    return Tf, Rv0, Rh0


def RaV_integration_function(Zy, Zx, iza, sigx, sigy, eps):
    A = np.cos(iza + 0.01) + Zx * np.sin(iza + 0.01)
    B = eps * (1 + Zx ** 2 + Zy ** 2)

    CC = np.sin(iza + 0.01) ** 2 - 2 * Zx * np.sin(iza + 0.01) * np.cos(iza + 0.01) + Zx ** 2 * np.cos(
        iza + 0.01) ** 2 + Zy ** 2

    Rv = (eps * A - np.sqrt(B - CC)) / (
            eps * A + np.sqrt(B - CC))

    pd = np.exp(-Zx ** 2 / (2 * sigx ** 2) - Zy ** 2 / (2 * sigy ** 2))

    Rav = Rv * pd

    return Rav


def RaH_integration_function(Zy, Zx, iza, sigx, sigy, eps):
    A = np.cos(iza + 0.01) + Zx * np.sin(iza + 0.01)
    B = eps * (1 + Zx ** 2 + Zy ** 2)

    CC = np.sin(iza + 0.01) ** 2 - 2 * Zx * np.sin(iza + 0.01) * np.cos(iza + 0.01) + Zx ** 2 * np.cos(
        iza + 0.01) ** 2 + Zy ** 2

    Rh = (A - np.sqrt(B - CC)) / (A + np.sqrt(B - CC))

    pd = np.exp(-Zx ** 2 / (2 * sigx ** 2) - Zy ** 2.0 / (2 * sigy ** 2))

    RaH = Rh * pd

    return RaH


def RaV_integration(iza, sigx, sigy, eps):
    bound = 3 * sigx

    rav = []
    for i in srange(len(iza)):
        ravv = dblquad(RaV_integration_function, -bound, bound, lambda x: -bound, lambda x: bound,
                       args=(iza[i], sigx, sigy, eps))
        temp = np.asarray(ravv[0]) / (2 * np.pi * sigx * sigy)
        rav.append(temp)

    Rav = np.asarray(rav)

    return Rav


def RaH_integration(iza, sigx, sigy, eps):
    bound = 3 * sigx

    rah = []
    for i in srange(len(iza)):
        rahh = dblquad(RaH_integration_function, -bound, bound, lambda x: -bound, lambda x: bound,
                       args=(iza[i], sigx, sigy, eps))
        temp = np.asarray(rahh[0]) / (2 * np.pi * sigx * sigy)
        rah.append(temp)

    RaH = np.asarray(rah)

    return RaH


def Ra_integration(iza, sigma, corrlength, eps):
    sigx = 1.1 * sigma / corrlength
    sigy = sigx

    RaV = RaV_integration(iza, sigx, sigy, eps)
    RaH = RaH_integration(iza, sigx, sigy, eps)

    return RaV, RaH


def biStatic_coefficient(iza, vza, raa, Rvi, Rv0, Rhi, Rh0, RaV, RaH, Tf):
    if np.array_equal(vza, iza) and (np.allclose(np.all(raa), np.pi)):
        Rvt = Rvi + (Rv0 - Rvi) * Tf
        Rht = Rhi + (Rh0 - Rhi) * Tf

    else:
        Rvt = RaV
        Rht = RaH

    fvv = 2 * Rvt * (np.sin(iza + 0.01) * np.sin(vza) - (1 + np.cos(iza + 0.01) * np.cos(vza)) * np.cos(raa)) / (
            np.cos(iza + 0.01) + np.cos(vza))

    fhh = -2 * Rht * (np.sin(iza + 0.01) * np.sin(vza) - (1 + np.cos(iza + 0.01) * np.cos(vza)) * np.cos(raa)) / (
            np.cos(iza + 0.01) + np.cos(vza))

    return fvv, fhh, Rvt, Rht


def Fppupdn_calc(ud, method, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs):
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


def Ipp(iza, vza, raa, phi, Rvi, Rhi, eps, k, kz_iza, kz_vza, fvv, fhh, sigma, Ts):
    Fvvupi, Fhhupi = Fppupdn_calc(+1, 1,
                                  Rvi,
                                  Rhi,
                                  eps,
                                  k,
                                  kz_iza,
                                  kz_vza,
                                  np.sin(iza + 0.01),
                                  np.cos(iza + 0.01),
                                  np.sin(vza),
                                  np.cos(vza),
                                  np.cos(phi),
                                  np.cos(raa),
                                  np.sin(raa))

    Fvvups, Fhhups = Fppupdn_calc(+1, 2,
                                  Rvi,
                                  Rhi,
                                  eps,
                                  k,
                                  kz_iza,
                                  kz_vza,
                                  np.sin(iza + 0.01),
                                  np.cos(iza + 0.01),
                                  np.sin(vza),
                                  np.cos(vza),
                                  np.cos(phi),
                                  np.cos(raa),
                                  np.sin(raa))

    Fvvdni, Fhhdni = Fppupdn_calc(-1, 1,
                                  Rvi,
                                  Rhi,
                                  eps,
                                  k,
                                  kz_iza,
                                  kz_vza,
                                  np.sin(iza + 0.01),
                                  np.cos(iza + 0.01),
                                  np.sin(vza),
                                  np.cos(vza),
                                  np.cos(phi),
                                  np.cos(raa),
                                  np.sin(raa))

    Fvvdns, Fhhdns = Fppupdn_calc(-1, 2,
                                  Rvi,
                                  Rhi,
                                  eps,
                                  k,
                                  kz_iza,
                                  kz_vza,
                                  np.sin(iza + 0.01),
                                  np.cos(iza + 0.01),
                                  np.sin(vza),
                                  np.cos(vza),
                                  np.cos(phi),
                                  np.cos(raa),
                                  np.sin(raa))

    qi = k * np.cos(iza + 0.01)
    qs = k * np.cos(vza)

    ivv = []
    ihh = []
    for i in srange(Ts):
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
    Ivv = np.asarray(ivv, dtype=np.complex)
    Ihh = np.asarray(ihh, dtype=np.complex)

    return Ivv, Ihh


def shadowing_function(iza, vza, raa, rss):
    if np.array_equal(vza, iza) and (np.allclose(np.all(raa), np.pi)):
        ct = cot(iza)
        cts = cot(vza)
        rslp = rss
        ctorslp = ct / np.sqrt(2) / rslp
        ctsorslp = cts / np.sqrt(2) / rslp
        shadf = 0.5 * (np.exp(-ctorslp ** 2) / np.sqrt(np.pi) / ctorslp - erf(ctorslp))
        shadfs = 0.5 * (np.exp(-ctsorslp ** 2) / np.sqrt(np.pi) / ctsorslp - erf(ctsorslp))
        ShdwS = 1 / (1 + shadf + shadfs)
    else:
        ShdwS = 1

    return ShdwS


def emsv_integralfunc(x, y, iza, eps, rv, rh, k, kl, ks, sq, corrfunc, corrlength, pol):
    error = 1.0e3

    sqs = np.sqrt(eps - np.sin(x) ** 2)
    rc = (rv - rh) / 2
    tv = 1 + rv
    th = 1 + rh

    # -- calc coefficients for surface correlation spectra
    wvnb = k * np.sqrt(
        np.sin(iza) ** 2 - 2 * np.sin(iza) * np.sin(x) * np.cos(y) + np.sin(x) ** 2)

    try:
        nr = len(x)

    except (IndexError, TypeError):
        nr = 1

    # -- calculate number of spectral components needed
    n_spec = 1
    while error > 1.0e-3:
        n_spec = n_spec + 1
        #   error = (ks2 *(cs + css)**2 )**n_spec / factorial(n_spec)
        # ---- in this case we will use the smallest ths to determine the number of
        # spectral components to use.  It might be more than needed for other angles
        # but this is fine.  This option is used to simplify calculations.
        error = (ks ** 2 * (np.cos(iza) + np.cos(x)) ** 2) ** n_spec / factorial(n_spec)
        error = np.min(error)
    # -- calculate expressions for the surface spectra
    wn = corrfunc(n_spec, nr, wvnb, kl, corrlength)

    # -- calculate fpq!

    ff = 2 * (np.sin(iza) * np.sin(x) - (1 + np.cos(iza) * np.cos(x)) * np.cos(y)) / (
            np.cos(iza) + np.cos(x))

    fvv = rv * ff
    fhh = -rh * ff

    fvh = -2 * rc * np.sin(y)
    fhv = 2 * rc * np.sin(y)

    # -- calculate Fpq and Fpqs -----
    fhv = np.sin(iza) * (np.sin(x) - np.sin(iza) * np.cos(y)) / (np.cos(iza) ** 2 * np.cos(x))
    T = (sq * (np.cos(iza) + sq) + np.cos(iza) * (
            eps * np.cos(iza) + sq)) / (
                eps * np.cos(iza) * (np.cos(iza) + sq) + sq * (
                eps * np.cos(iza) + sq))
    cm2 = np.cos(x) * sq / np.cos(iza) / sqs - 1
    ex = np.exp(-ks ** 2 * np.cos(iza) * np.cos(x))
    de = 0.5 * np.exp(-ks ** 2 * (np.cos(iza) ** 2 + np.cos(x) ** 2))

    if pol == 'vv':
        Fvv = (eps - 1) * np.sin(iza) ** 2 * tv ** 2 * fhv / eps ** 2
        Fhv = (T * np.sin(iza) * np.sin(iza) - 1. + np.cos(iza) / np.cos(x) + (
                eps * T * np.cos(iza) * np.cos(x) * (
                eps * T - np.sin(iza) * np.sin(iza)) - sq * sq) / (
                       T * eps * sq * np.cos(x))) * (1 - rc * rc) * np.sin(y)

        Fvvs = -cm2 * sq * tv ** 2 * (
                np.cos(y) - np.sin(iza) * np.sin(x)) / np.cos(
            iza) ** 2 / eps - cm2 * sqs * tv ** 2 * np.cos(y) / eps - (
                       np.cos(x) * sq / np.cos(iza) / sqs / eps - 1) * np.sin(
            x) * tv ** 2 * (
                       np.sin(iza) - np.sin(x) * np.cos(y)) / np.cos(iza)
        Fhvs = -(np.sin(x) * np.sin(x) / T - 1 + np.cos(x) / np.cos(iza) + (
                np.cos(iza) * np.cos(x) * (
                1 - np.sin(x) * np.sin(x) * T) - T * T * sqs * sqs) / (
                         T * sqs * np.cos(iza))) * (1 - rc * rc) * np.sin(y)

        # -- calculate the bistatic field coefficients ---

        svv = np.zeros([n_spec, nr])
        for n in srange(n_spec):
            Ivv = fvv * ex * (ks * (np.cos(iza) + np.cos(x))) ** (n + 1) + (
                    Fvv * (ks * np.cos(x)) ** (n + 1) + Fvvs * (ks * np.cos(iza)) ** (n + 1)) / 2
            Ihv = fhv * ex * (ks * (np.cos(iza) + np.cos(x))) ** (n + 1) + (
                    Fhv * (ks * np.cos(x)) ** (n + 1) + Fhvs * (ks * np.cos(iza)) ** (n + 1)) / 2

        wnn = wn[n, :] / factorial(n + 1)
        vv = wnn * (abs(Ivv)) ** 2
        hv = wnn * (abs(Ihv)) ** 2
        svv[n, :] = (de * (vv + hv) * np.sin(x) * (1 / np.cos(iza))) / (4 * np.pi)

        ref = np.sum([svv])  # adding all n terms stores in different rows

    if pol == 'hh':
        Fhh = -(eps - 1) * th ** 2 * fhv
        Fvh = (np.sin(iza) * np.sin(iza) / T - 1. + np.cos(iza) / np.cos(x) + (
                np.cos(iza) * np.cos(x) * (
                1 - np.sin(iza) * np.sin(iza) * T) - T * T * sq * sq) / (
                       T * sq * np.cos(x))) * (1 - rc * rc) * np.sin(y)

        Fhhs = cm2 * sq * th ** 2 * (
                np.cos(y) - np.sin(iza) * np.sin(x)) / np.cos(
            iza) ** 2 + cm2 * sqs * th ** 2 * np.cos(y) + cm2 * np.sin(x) * th ** 2 * (
                       np.sin(iza) - np.sin(x) * np.cos(y)) / np.cos(iza)
        Fvhs = -(T * np.sin(x) * np.sin(x) - 1 + np.cos(x) / np.cos(iza) + (
                eps * T * np.cos(iza) * np.cos(x) * (
                eps * T - np.sin(x) * np.sin(x)) - sqs * sqs) / (
                         T * eps * sqs * np.cos(iza))) * (1 - rc * rc) * np.sin(y)

        shh = np.zeros([n_spec, nr])
        for n in srange(n_spec):
            Ihh = fhh * ex * (ks * (np.cos(iza) + np.cos(x))) ** (n + 1) + (
                    Fhh * (ks * np.cos(x)) ** (n + 1) + Fhhs * (ks * np.cos(iza)) ** (n + 1)) / 2
            Ivh = fvh * ex * (ks * (np.cos(iza) + np.cos(x))) ** (n + 1) + (
                    Fvh * (ks * np.cos(x)) ** (n + 1) + Fvhs * (ks * np.cos(iza)) ** (n + 1)) / 2

        wnn = wn[n, :] / factorial(n + 1)
        hh = wnn * (abs(Ihh)) ** 2
        vh = wnn * (abs(Ivh)) ** 2
        (2 * (3 + 4) * np.sin(5) * 1 / np.cos(6)) / (np.pi * 4)
        shh[n, :] = (de * (hh + vh) * np.sin(x) * (1 / np.cos(iza))) / (4 * np.pi)

        ref = np.sum([shh])

    return ref
