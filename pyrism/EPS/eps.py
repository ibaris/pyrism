# -*- coding: utf-8 -*-
from __future__ import division

import sys

import numpy as np

# python 3.6 comparability
if sys.version_info < (3, 0):
    srange = xrange
else:
    srange = range


def water(frequency, temp):
    # <Help and Info Section> -----------------------------------------
    """
    Relative Dielectric Constant of Pure Water.
    Computes the real and imaginary parts of the relative
    dielectric constant of water at any temperature 0<t<30 and frequency
    0<f<1000 GHz. Uses the double-Debye model.

    Parameters
    ----------
    frequency : int, float or array_like
        Frequency (GHz).
    temp : int, float or array
        Temperature in C° (0 - 30).

    Returns
    -------
    Dielectric Constant:    complex

    """
    a = [0.63000075e1, 0.26242021e-2, 0.17667420e-3, 0.58366888e3, 0.12634992e3, 0.69227972e-4, 0.30742330e3,
         0.12634992e3, 0.37245044e1, 0.92609781e-2]

    epsS = 87.85306 * np.exp(-0.00456992 * temp)
    epsOne = a[0] * np.exp(-a[1] * temp)
    tau1 = a[2] * np.exp(a[3] / (temp + a[4]))
    tau2 = a[5] * np.exp(a[6] / (temp + a[7]))
    epsInf = a[8] + a[9] * temp

    eps = ((epsS - epsOne) / (1 - 1j * 2 * np.pi * frequency * tau1)) + (
            (epsOne - epsInf) / (1 - 1j * 2 * np.pi * frequency * tau2)) + epsInf

    return eps


def saline_water(frequency, temp, salinity):
    # <Help and Info Section> -----------------------------------------
    """
    Relative Dielectric Constant of Saline Water.
    Computes the real and imaginary parts of the relative
    dielectric constant of water at any temperature 0<t<30, Salinity
    0<Salinity<40 0/00, and frequency 0<f<1000GHz

    Parameters
    ----------
    frequency : int, float or array_like
        Frequency (GHz).
    temp : int, float or array
        Temperature in C° (0 - 30).
    salinity : int, float or array
        Salinity in parts per thousand.

    Returns
    -------
    Dielectric Constant:    complex

    """
    # Conductvity
    A = [2.903602, 8.607e-2, 4.738817e-4, -2.991e-6, 4.3041e-9]
    sig35 = A[0] + A[1] * temp + A[2] * temp ** 2 + A[3] * temp ** 3 + A[4] * temp ** 4

    A = [37.5109, 5.45216, 0.014409, 1004.75, 182.283]
    P = salinity * ((A[0] + A[1] * salinity + A[2] * salinity ** 2) / (A[3] + A[4] * salinity + salinity ** 2))

    A = [6.9431, 3.2841, -0.099486, 84.85, 69.024]
    alpha0 = (A[0] + A[1] * salinity + A[2] * salinity ** 2) / (A[3] + A[4] * salinity + salinity ** 2)

    A = [49.843, -0.2276, 0.00198]
    alpha1 = A[0] + A[1] * salinity + A[2] * salinity ** 2

    Q = 1 + ((alpha0 * (temp - 15)) / (temp + alpha1))

    sigma = sig35 * P * Q

    a = [0.46606917e-2, -0.26087876e-4, -0.63926782e-5, 0.63000075e1, 0.26242021e-2, -0.42984155e-2, 0.34414691e-4,
         0.17667420e-3, -0.20491560e-6, 0.58366888e3, 0.12634992e3, 0.69227972e-4, 0.38957681e-6, 0.30742330e3,
         0.12634992e3, 0.37245044e1, 0.92609781e-2, -0.26093754e-1]

    epsS = 87.85306 * np.exp(-0.00456992 * temp - a[0] * salinity - a[1] * salinity ** 2 - a[2] * salinity * temp)
    epsOne = a[3] * np.exp(-a[4] * temp - a[5] * salinity - a[6] * salinity * temp)
    tau1 = (a[7] + a[8] * salinity) * np.exp(a[9] / (temp + a[10]))
    tau2 = (a[11] + a[12] * salinity) * np.exp(a[13] / (temp + a[14]))
    epsInf = a[15] + a[16] * temp + a[17] * salinity

    eps = ((epsS - epsOne) / (1 - 1j * 2 * np.pi * frequency * tau1)) + (
            (epsOne - epsInf) / (1 - 1j * 2 * np.pi * frequency * tau2)) + epsInf + 1j * (
                  (17.9751 * sigma) / frequency)

    return eps


def soil(frequency, temp, S, C, mv, rho_b=1.7):
    # <Help and Info Section> -----------------------------------------
    """
    Relative Dielectric Constant of soil.
    Computes the real and imaginary parts of the relative
    dielectric constant of soil at a given temperature 0<t<40C, frequency,
    volumetric moisture content, soil bulk density, sand and clay
    fractions.

    Parameters
    ----------
    frequency : int, float or array_like
        Frequency (GHz).
    temp : int, float or array
        Temperature in C° (0 - 30).
    S : int or float
        Sand fraction in %.
    C : int or float
        Clay fraction in %.
    mv : int or float
        Volumetric Water Content (0<mv<1)
    rho_b : int or float (default = 1.7)
        Bulk density in g/cm3 (typical value is 1.7 g/cm3).

    Returns
    -------
    Dielectric Constant:    complex

    """
    frequency = np.asarray(frequency).flatten()
    epsl = []
    for i in srange(len(frequency)):
        f_hz = frequency[i] * 1.0e9

        beta1 = 1.27 - 0.519 * S - 0.152 * C
        beta2 = 2.06 - 0.928 * S - 0.255 * C
        alpha = 0.65

        eps_0 = 8.854e-12

        sigma_s = 0
        if frequency[i] > 1.3:
            sigma_s = -1.645 + 1.939 * rho_b - 2.256 * S + 1.594 * C

        if frequency[i] >= 0.3 and frequency[i] <= 1.3:
            sigma_s = 0.0467 + 0.22 * rho_b - 0.411 * S + 0.661 * C

        ew_inf = 4.9
        ew_0 = 88.045 - 0.4147 * temp + 6.295e-4 * temp ** 2 + 1.075e-5 * temp ** 3
        tau_w = (1.1109e-10 - 3.824e-12 * temp + 6.938e-14 * temp ** 2 - 5.096e-16 * temp ** 3) / 2 / np.pi

        epsrW = ew_inf + (ew_0 - ew_inf) / (1 + (2 * np.pi * f_hz * tau_w) ** 2)

        epsiW = 2 * np.pi * tau_w * f_hz * (ew_0 - ew_inf) / (1 + (2 * np.pi * f_hz * tau_w) ** 2) + (
                2.65 - rho_b) / 2.65 / mv * sigma_s / (2 * np.pi * eps_0 * f_hz)

        epsr = (1 + 0.66 * rho_b + mv ** beta1 * epsrW ** alpha - mv) ** (1 / alpha)
        epsi = mv ** beta2 * epsiW

        eps = np.complex(epsr, epsi)
        epsl.append(eps)

    return np.asarray(epsl, dtype=np.complex)


def vegetation(frequency, mg):
    # <Help and Info Section> -----------------------------------------
    """
    Relative Dielectric Constant of Vegetation.
    Computes the real and imaginary parts of the relative
    dielectric constant of vegetation material, such as corn leaves, in
    the microwave region.

    Parameters
    ----------
    frequency : int, float or array_like
        Frequency (GHz).
    mg : int or float
        Gravimetric moisture content (0<mg< 1).

    Returns
    -------
    Dielectric Constant:    complex

    """
    frequency = np.asarray(frequency).flatten()

    S = 15

    epsl = []
    for i in srange(len(frequency)):
        # free water in leaves
        sigma_i = 0.17 * S - 0.0013 * S ** 2

        eps_w_r = 4.9 + 74.4 / (1 + (frequency[i] / 18) ** 2)
        eps_w_i = 74.4 * (frequency[i] / 18) / (1 + (frequency[i] / 18) ** 2) + 18 * sigma_i / frequency[i]

        # bound water in leaves
        eps_b_r = 2.9 + 55 * (1 + np.sqrt(frequency[i] / 0.36)) / (
                (1 + np.sqrt(frequency[i] / 0.36)) ** 2 + (frequency[i] / 0.36))
        eps_b_i = 55 * np.sqrt(frequency[i] / 0.36) / (
                (1 + np.sqrt(frequency[i] / 0.36)) ** 2 + (frequency[i] / 0.36))

        # emnp.pirical fits
        v_fw = mg * (0.55 * mg - 0.076)
        v_bw = 4.64 * mg ** 2 / (1 + 7.36 * mg ** 2)

        eps_r = 1.7 - 0.74 * mg + 6.16 * mg ** 2
        eps_v_r = eps_r + v_fw * eps_w_r + v_bw * eps_b_r
        eps_v_i = v_fw * eps_w_i + v_bw * eps_b_i

        eps = np.complex(eps_v_r, eps_v_i)
        epsl.append(eps)

    return np.asarray(epsl, dtype=np.complex)
