# cython: cdivision=True
from __future__ import division

cimport numpy as np
import numpy as np
import cmath
import scipy
from scipy.integrate import dblquad
import cython
from scipy.special import erf
from libc.math cimport sin, cos, pow, sqrt, exp
import sys

DTYPE = np.float

ctypedef np.float_t DTYPE_t

cdef:
    double PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164

# Compute I2EM Backscattering ##########################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# Auxiliary Function
# ----------------------------------------------------------------------------------------------------------------------
cdef int factorial(int x):
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

# ----------------------------------------------------------------------------------------------------------------------
# Computation of Rx0 and Rxi
# ----------------------------------------------------------------------------------------------------------------------
cdef tuple compute_Rx0(double complex[:] eps):
    cdef:
        Py_ssize_t xmax = eps.shape[0]
        Py_ssize_t i
        double complex[:] Rv0_view, Rh0_view

    Rv0 = np.zeros_like(eps, dtype=np.complex)
    Rh0 = np.zeros_like(eps, dtype=np.complex)
    Rv0_view = Rv0
    Rh0_view = Rh0

    for i in range(xmax):
        Rv0_view[i] = (cmath.sqrt(eps[i]) - 1) / (cmath.sqrt(eps[i]) + 1)
        Rh0_view[i] = -Rv0_view[i]

    return Rv0, Rh0

cdef tuple compute_Rxi(double[:] iza, double complex[:] eps, double complex[:] rt):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double complex[:] Rvi_view, Rhi_view
        double mui

    Rvi = np.zeros_like(iza, dtype=np.complex)
    Rhi = np.zeros_like(iza, dtype=np.complex)
    Rvi_view = Rvi
    Rhi_view = Rhi

    for i in range(xmax):
        mui = cos(iza[i])
        Rvi_view[i] = (eps[i] * mui - rt[i]) / (eps[i] * mui + rt[i])
        Rhi_view[i] = (mui - rt[i]) / (mui + rt[i])

    return Rvi, Rhi

# ----------------------------------------------------------------------------------------------------------------------
# Computation of Wn, rss, RT, WVNB, Ts and Ft
# ----------------------------------------------------------------------------------------------------------------------
cdef tuple compute_Wn_rss(corrfunc, double[:] iza, double[:] vza, double[:] raa, double[:] phi, double[:] k,
                          double[:] sigma, double[:] corrlength, int[:] n):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i, j

        double rss_temp
        double[:, :] Wn_view
        double[:] rss_view, wvnb, Wn_temp
        int[:] Ts

    wvnb = compute_wvnb(iza=iza, vza=vza, raa=raa, phi=phi, k=k)
    Ts = compute_TS(iza=iza, vza=vza, sigma=sigma, k=k)

    Wn = np.zeros((xmax, max(Ts.base)))
    rss = np.zeros_like(iza)

    Wn_view = Wn
    rss_view = rss

    for i in range(xmax):
        Wn_temp, rss_temp = corrfunc(sigma[i], corrlength[i], wvnb[i], Ts[i])

        for j in range(Wn_temp.shape[0]):
            Wn_view[i, j] = Wn_temp[j]

        rss_view[i] = rss_temp

    return Wn, rss

cdef double complex[:] compute_rt(double[:] iza, double[:] epsr, double[:] epsi):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double complex[:] rt_view
        double complex eps
        double temp

    rt = np.zeros_like(iza, dtype=np.complex)
    rt_view = rt

    for i in range(xmax):
        eps = complex(epsr[i], epsi[i])
        rt[i] = cmath.sqrt(eps - pow(sin(iza[i]), 2))
        # rt[i] = complex(temp, epsi[i])

    return rt

cdef double[:] compute_wvnb(double[:] iza, double[:] vza, double[:] raa, double[:] phi, double[:] k):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double[:] wvnb_view
        double sin_vza, sin_iza, cos_raa, cos_phi, sin_raa, sin_phi, part1, part2


    wvnb = np.zeros_like(iza, dtype=np.double)
    wvnb_view = wvnb

    for i in range(xmax):
        sin_vza = sin(vza[i])
        cos_raa = cos(raa[i])
        sin_iza = sin(iza[i])
        cos_phi = cos(phi[i])
        sin_raa = sin(raa[i])
        sin_phi = sin(phi[i])

        part1 = pow((sin_vza * cos_raa - sin_iza * cos_phi), 2)
        part2 = pow((sin_vza * sin_raa - sin_iza * sin_phi), 2)

        wvnb_view[i] = k[i] * sqrt(part1 + part2)

    return wvnb

def wvnb_wrapper(double[:] iza, double[:] vza, double[:] raa, double[:] phi, double[:] k):
    return compute_wvnb(iza, vza, raa, phi, k)

cdef int[:] compute_TS(double[:] iza, double[:] vza, double[:] sigma, double[:] k):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        int[:] TS_view
        int TS_temp
        double mui, muv, merror, error

    TS = np.zeros_like(iza, dtype=np.intc)
    TS_view = TS

    for i in range(xmax):
        mui = cos(iza[i])
        muv = cos(vza[i])
        TS_temp = 1
        error = 1.0

        while error > 1.0e-8: # and TS_temp <= 150:
            TS_temp += 1
            error = pow(pow(k[i] * sigma[i], 2) * pow((mui + muv), 2), TS_temp) / factorial(TS_temp)

        TS_view[i] = TS_temp

    return TS

def TS_wrapper(double[:] iza, double[:] vza, double[:] sigma, double[:] k):
    return compute_TS(iza, vza, sigma, k)

cdef double complex[:] compute_Ft(double[:] iza, double[:] vza, double complex[:] eps):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double complex[:] Rv0, Rh0, Ft_view
        double sin_iza, sin_vza, cos_iza
        double complex Rv02

    Rv0, Rh0 = compute_Rx0(eps)

    Ft = np.zeros_like(iza, dtype=np.complex)
    Ft_view = Ft

    for i in range(xmax):
        sin_iza = sin(iza[i])
        sin_vza = sin(vza[i])
        cos_iza = cos(iza[i])
        Rv02 = Rv0[i] * Rv0[i]

        Ft_view[i] = 8 * Rv02 * sin_vza * (cos_iza + cmath.sqrt(eps[i] - pow(sin_iza, 2))) / (cos_iza * cmath.sqrt(eps[i] - pow(sin_iza, 2)))

    return Ft

cdef double[:] compute_Tf(double[:] iza, double[:] k, double[:] sigma, double complex[:] Rv0, double complex[:] Ft,
                          double[:, :] Wn, int[:] Ts):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t x, i, index
        double a0, a1, b1, temp, St, St0
        double[:] Tf_view

    Tf = np.zeros_like(iza, dtype=np.double)
    Tf_view = Tf

    a1, b1 = 0.0, 0.0

    for i in range(xmax):
        for x in range(1, Ts[i] + 1):
            index = x - 1
            cos_iza = cos(iza[i])

            a0 = pow(k[i] * sigma[i] * cos_iza, 2*x) / factorial(x)
            a1 += a0 * Wn[i, index]

            temp = abs(Ft[i] / 2 + pow(2, x+1) * Rv0[i] / cos_iza * exp(- pow(k[i] * sigma[i] * cos_iza, 2)))
            b1 += a0 * pow(temp, 2) * Wn[i, index]

        St = 0.25 * pow(abs(Ft[i]), 2) * a1 / b1
        St0 = 1 / pow(abs(1 + 8 * Rv0[i] / (cos_iza * Ft[i])), 2)

        Tf_view[i] = 1-St/St0

    return Tf

# ----------------------------------------------------------------------------------------------------------------------
# Integration
# ----------------------------------------------------------------------------------------------------------------------
# Auxiliary Functions ----------------------------------------------------------------------------------------------
cdef tuple compute_ABCC(double Zy, double Zx, double iza, double complex eps):
    cdef:
        double A, CC, cos_iza, sin_iza, pd
        double complex B

    cos_iza = cos(iza)
    sin_iza = sin(iza)

    A = cos_iza + Zx * sin_iza
    B = (1 + pow(Zx, 2) + pow(Zy, 2)) * eps
    CC = pow(sin_iza, 2) - 2 * Zx * sin_iza * cos_iza + pow(Zx, 2) * pow(cos_iza, 2) + pow(Zy, 2)

    return A, B, CC

# Callable Integration Functions -----------------------------------------------------------------------------------
cdef double complex RaV_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps):
    cdef:
        double A, CC, pd
        double complex B, Rv, Rav

    A, B, CC = compute_ABCC(Zy, Zx, iza, eps)
    pd = exp(-pow(Zx, 2) / (2 * pow(sigx, 2)) - pow(Zy, 2) / (2 * pow(sigy, 2)))

    Rv = (eps * A - cmath.sqrt(B - CC)) / (eps * A + cmath.sqrt(B - CC))
    Rav = Rv * pd

    return Rav#.real

cdef double complex RaH_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps):
    cdef:
        double A, CC, pd
        double complex B, Rh, Rah

    A, B, CC = compute_ABCC(Zy, Zx, iza, eps)
    pd = exp(-pow(Zx, 2) / (2 * pow(sigx, 2)) - pow(Zy, 2) / (2 * pow(sigy, 2)))

    Rh = (A - cmath.sqrt(B - CC)) / (A + cmath.sqrt(B - CC))
    RaH = Rh * pd

    return RaH#.real

cdef real_RaV_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps):
    return scipy.real(RaV_integration_ifunc(Zy, Zx, iza, sigx, sigy, eps))

cdef imag_RaV_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps):
    return scipy.imag(RaV_integration_ifunc(Zy, Zx, iza, sigx, sigy, eps))

cdef real_RaH_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps):
    return scipy.real(RaH_integration_ifunc(Zy, Zx, iza, sigx, sigy, eps))

cdef imag_RaH_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps):
    return scipy.imag(RaH_integration_ifunc(Zy, Zx, iza, sigx, sigy, eps))

# Integration with SciPy -------------------------------------------------------------------------------------------
cdef tuple Rax_integration(double[:] iza, double[:] sigma, double[:] corrlength, double complex[:] eps):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double bound, ravv, rahh
        double complex[:] Rav_view, Rah_view
        double sigy, sigx

    Rav = np.zeros_like(iza, dtype=np.complex)
    Rah = np.zeros_like(iza, dtype=np.complex)

    Rav_view = Rav
    Rah_view = Rah

    for i in range(xmax):
        sigx = 1.1 * sigma[i] / corrlength[i]
        sigy = sigx

        bound = 3 * sigx

        ravv_real = dblquad(real_RaV_integration_ifunc, -bound, bound, lambda x: -bound, lambda x: bound,
                   args=(iza[i], sigx, sigy, eps[i]))[0]

        rahh_real = dblquad(real_RaH_integration_ifunc, -bound, bound, lambda x: -bound, lambda x: bound,
                   args=(iza[i], sigx, sigy, eps[i]))[0]

        ravv_imag = dblquad(imag_RaV_integration_ifunc, -bound, bound, lambda x: -bound, lambda x: bound,
                   args=(iza[i], sigx, sigy, eps[i]))[0]

        rahh_imag = dblquad(imag_RaH_integration_ifunc, -bound, bound, lambda x: -bound, lambda x: bound,
                   args=(iza[i], sigx, sigy, eps[i]))[0]


        # ravv = dblquad(RaV_integration_ifunc, -bound, bound, lambda x: -bound, lambda x: bound,
        #            args=(iza[i], sigx, sigy, eps[i]))[0]
        #
        # rahh = dblquad(RaH_integration_ifunc, -bound, bound, lambda x: -bound, lambda x: bound,
        #            args=(iza[i], sigx, sigy, eps[i]))[0]

        Rav_view[i] = complex(ravv_real, ravv_imag) / (2 * PI * sigx * sigy)
        Rah_view[i] = complex(rahh_real, rahh_imag) / (2 * PI * sigx * sigy)

    return Rav, Rah

# ----------------------------------------------------------------------------------------------------------------------
# Computation of Bistatic Coefficients
# ----------------------------------------------------------------------------------------------------------------------
# Compute Rvt and Rht ----------------------------------------------------------------------------------------------
cdef tuple compute_Rxt(double[:] iza, double[:] vza, double[:] raa, double[:] sigma, double[:] corrlength,
                       double complex[:] eps, double[:] Tf):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double complex[:] RaV, RaH
        double complex[:] Rvt_view, Rht_view, rt, Rv0, Rh0, Rvi, Rhi

    rt = compute_rt(iza, eps.base.real, eps.base.imag)

    Rv0, Rh0 = compute_Rx0(eps)
    Rvi, Rhi = compute_Rxi(iza, eps, rt)

    Rvt = np.zeros_like(iza, dtype=np.complex)
    Rht = np.zeros_like(iza, dtype=np.complex)

    RaV, RaH = Rax_integration(iza, sigma, corrlength, eps)

    Rvt_view = Rvt
    Rht_view = Rht

    for i in range(xmax):
        if vza[i] == iza[i] and np.allclose(raa[i], PI):
            Rvt_view[i] = Rvi[i] + (Rv0[i] - Rvi[i]) * Tf[i]
            Rht_view[i] = Rhi[i] + (Rh0[i] - Rhi[i]) * Tf[i]

        else:
            Rvt_view[i] = RaV[i]
            Rht_view[i] = RaH[i]

    return Rvt, Rht

# Compute fvv and fhh ----------------------------------------------------------------------------------------------
cdef tuple compute_fxx(double complex[:] Rvt, double complex[:] Rht, double[:] iza, double[:] vza, double[:] raa):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double temp
        double complex[:] fvv_view, fhh_view

    fvv = np.zeros_like(Rvt, dtype=np.complex)
    fhh = np.zeros_like(Rht, dtype=np.complex)

    fvv_view = fvv
    fhh_view = fhh

    for i in range(xmax):
        temp = ((sin(iza[i]) * sin(vza[i]) - (1 + np.cos(iza[i]) * np.cos(vza[i])) *
                np.cos(raa[i])) / (np.cos(iza[i]) + np.cos(vza[i])))

        fvv_view[i] = 2 * Rvt[i] * temp
        fhh_view[i] = -2 * Rht[i] * temp

    return fvv, fhh


# ----------------------------------------------------------------------------------------------------------------------
# Comutation of Fxxuxxx
# ----------------------------------------------------------------------------------------------------------------------
# Compute Fxx with Method One --------------------------------------------------------------------------------------
cdef tuple compute_Cm1(int ud, double complex[:] Rvi, double complex[:] Rhi, double complex[:] eps, double[:]
                       k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                       double[:] phi):

    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i

        # < Temporaly Variable Definitions > ------------
        double Gqi
        double complex Gqti, qt

        double sin_iza, sin_vza, sin_raa, sin_phi, ki, k2
        double cos_iza, cos_vza, cos_raa, cos_phi

        # < Definitions of C Parameter > ------------
        double c11, c12, c21, c31, c41, c42, c51
        double complex c22, c32, c52, Rvii, Rhii

        # < Definition of Fxx > ------------
        double complex[:] Fvv_view, Fhh_view


    Fvv = np.zeros_like(iza, dtype=np.complex)
    Fhh = np.zeros_like(Fvv, dtype=np.complex)

    Fvv_view = Fvv
    Fhh_view = Fhh

    for i in range(xmax):
        Rvii = Rvi[i]
        Rhii = Rhi[i]

        sin_iza = sin(iza[i])
        sin_vza = sin(vza[i])
        sin_raa = sin(raa[i])
        sin_phi = sin(phi[i])

        cos_iza = cos(iza[i])
        cos_vza = cos(vza[i])
        cos_raa = cos(raa[i])
        cos_phi = cos(phi[i])

        ki = k[i]
        k2 = pow(ki, 2)

        Gqi = ud * kz_iza[i]
        Gqti = ud * ki * cmath.sqrt(eps[i] - pow(sin_iza, 2))

        # < Matrix Elements > ------------
        c11 = ki * cos_raa * (kz_vza[i] - Gqi)
        c12 = c11

        c21 = cos_iza * (cos_raa *
                                 (k2 * sin_iza * cos_phi * (sin_vza * cos_raa - sin_iza * cos_phi) +
                                  Gqi * (ki * cos_vza - Gqi)) +
                                 k2 * cos_phi * sin_iza * sin_vza * pow(sin_raa, 2))

        c22 = cos_iza * (cos_raa *
                                 (k2 * sin_iza * cos_phi * (sin_vza * cos_raa - sin_iza * cos_phi) +
                                  Gqti * (ki * cos_vza - Gqi)) + k2 * cos_phi * sin_iza * sin_vza * pow(sin_raa, 2))

        c31 = ki * sin_iza * (sin_iza * cos_phi * cos_raa * (ki * cos_vza - Gqi) -
                                      Gqi * (cos_raa * (sin_vza * cos_raa - sin_iza * cos_phi) +
                                             sin_vza * pow(sin_raa, 2)))

        c32 = ki * sin_iza * (sin_iza * cos_phi * cos_raa * (ki * cos_vza - Gqi) -
                                      Gqti * (cos_raa * (sin_vza * cos_raa - sin_iza * cos_phi) -
                                              sin_vza * pow(sin_raa, 2)))

        c41 = ki * cos_iza * (cos_raa * cos_vza * (ki * cos_vza - Gqi) +
                                      ki * sin_vza * (sin_vza * cos_raa - sin_iza * cos_phi))
        c42 = c41

        c51 = Gqi * (cos_raa * cos_vza * (Gqi - ki * cos_vza) -
                             ki * sin_vza * (sin_vza * cos_raa - sin_iza * cos_phi))
        c52 = Gqti * (cos_raa * cos_vza * (Gqi - ki * cos_vza) -
                              ki * sin_vza * (sin_vza * cos_raa - sin_iza * cos_phi))


        qt = ki * cmath.sqrt(eps[i] - pow(sin_iza, 2))

        Fvv_view[i] = (1 + Rvii) * (-(1 - Rvii) * c11 / kz_iza[i] + (1 + Rvii) * c12 / qt) + \
                      (1 - Rvii) * ((1 - Rvii) * c21 / kz_iza[i] - (1 + Rvii) * c22 / qt) + \
                      (1 + Rvii) * ((1 - Rvii) * c31 / kz_iza[i] - (1 + Rvii) * c32 / eps[i] / qt) + \
                      (1 - Rvii) * ((1 + Rvii) * c41 / kz_iza[i] - eps[i] * (1 - Rvii) * c42 / qt) + \
                      (1 + Rvii) * ((1 + Rvii) * c51 / kz_iza[i] - (1 - Rvii) * c52 / qt)

        Fhh_view[i] = (1 + Rhii) * ((1 - Rhii) * c11 / kz_iza[i] - eps[i] * (1 + Rhii) * c12 / qt) - \
                      (1 - Rhii) * ((1 - Rhii) * c21 / kz_iza[i] - (1 + Rhii) * c22 / qt) - \
                      (1 + Rhii) * ((1 - Rhii) * c31 / kz_iza[i] - (1 + Rhii) * c32 / qt) - \
                      (1 - Rhii) * ((1 + Rhii) * c41 / kz_iza[i] - (1 - Rhii) * c42 / qt) - \
                      (1 + Rhii) * ((1 + Rhii) * c51 / kz_iza[i] - (1 - Rhii) * c52 / qt)

    return Fvv, Fhh

# Compute Fxx with Method Two --------------------------------------------------------------------------------------
cdef tuple compute_Cm2(int ud, double complex[:] Rvi, double complex[:] Rhi, double complex[:] eps, double[:]
                       k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                       double[:] phi):

    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i

        # < Temporaly Variable Definitions > ------------
        double Gqi
        double complex Gqti, qt

        double sin_iza, sin_vza, sin_raa, sin_phi, ki, k2
        double cos_iza, cos_vza, cos_raa, cos_phi

        # < Definitions of C Parameter > ------------
        double c11, c12, c21, c31, c41, c42, c51
        double complex c22, c32, c52, Rvii, Rhii

        # < Definition of Fxx > ------------
        double complex[:] Fvv_view, Fhh_view


    Fvv = np.zeros_like(iza, dtype=np.complex)
    Fhh = np.zeros_like(Fvv, dtype=np.complex)

    Fvv_view = Fvv
    Fhh_view = Fhh

    for i in range(xmax):
        Rvii = Rvi[i]
        Rhii = Rhi[i]

        sin_iza = sin(iza[i])
        sin_vza = sin(vza[i])
        sin_raa = sin(raa[i])
        sin_phi = sin(phi[i])

        cos_iza = cos(iza[i])
        cos_vza = cos(vza[i])
        cos_raa = cos(raa[i])
        cos_phi = cos(phi[i])

        ki = k[i]
        k2 = pow(ki, 2)

        Gqs = ud * kz_vza[i]
        Gqts = ud * ki * cmath.sqrt(eps[i] - pow(sin_vza, 2))

        # < Matrix Elements > ------------
        c11 = ki * cos_raa * (kz_iza[i] + Gqs)
        c21 = Gqs * (cos_raa * (cos_iza * (ki * cos_iza + Gqs) - ki * sin_iza * (
                    sin_vza * cos_raa - sin_iza * cos_phi)) - ki * sin_iza * sin_vza * sin_raa ** 2)
        c31 = ki * sin_vza * (ki * cos_iza * (sin_vza * cos_raa - sin_iza * cos_phi) + sin_iza * (kz_iza[i] + Gqs))
        c41 = ki * cos_vza * (cos_raa * (cos_iza * (kz_iza[i] + Gqs) - ki * sin_iza * (
                    sin_vza * cos_raa - sin_iza * cos_phi)) - ki * sin_iza * sin_vza * sin_raa ** 2)
        c51 = -cos_vza * (
                    ki ** 2 * sin_vza * (sin_vza * cos_raa - sin_iza * cos_phi) + Gqs * cos_raa * (kz_iza[i] + Gqs))
        
        c12 = ki * cos_raa * (kz_iza[i] + Gqs)
        c22 = Gqts * (cos_raa * (cos_iza * (kz_iza[i] + Gqs) - ki * sin_iza * (
                    sin_vza * cos_raa - sin_iza * cos_phi)) - ki * sin_iza * sin_vza * sin_raa ** 2)
        c32 = ki * sin_vza * (ki * cos_iza * (sin_vza * cos_raa - sin_iza * cos_phi) + sin_iza * (kz_iza[i] + Gqs))
        c42 = ki * cos_vza * (cos_raa * (cos_iza * (kz_iza[i] + Gqs) - ki * sin_iza * (
                    sin_vza * cos_raa - sin_iza * cos_phi)) - ki * sin_iza * sin_vza * sin_raa ** 2)
        c52 = -cos_vza * (
                    ki ** 2 * sin_vza * (sin_vza * cos_raa - sin_iza * cos_phi) + Gqts * cos_raa * (kz_iza[i] + Gqs))


        qt = ki * cmath.sqrt(eps[i] - pow(sin_iza, 2))

        Fvv_view[i] = (1 + Rvii) * (-(1 - Rvii) * c11 / kz_iza[i] + (1 + Rvii) * c12 / qt) + \
                      (1 - Rvii) * ((1 - Rvii) * c21 / kz_iza[i] - (1 + Rvii) * c22 / qt) + \
                      (1 + Rvii) * ((1 - Rvii) * c31 / kz_iza[i] - (1 + Rvii) * c32 / eps[i] / qt) + \
                      (1 - Rvii) * ((1 + Rvii) * c41 / kz_iza[i] - eps[i] * (1 - Rvii) * c42 / qt) + \
                      (1 + Rvii) * ((1 + Rvii) * c51 / kz_iza[i] - (1 - Rvii) * c52 / qt)

        Fhh_view[i] = (1 + Rhii) * ((1 - Rhii) * c11 / kz_iza[i] - eps[i] * (1 + Rhii) * c12 / qt) - \
                      (1 - Rhii) * ((1 - Rhii) * c21 / kz_iza[i] - (1 + Rhii) * c22 / qt) - \
                      (1 + Rhii) * ((1 - Rhii) * c31 / kz_iza[i] - (1 + Rhii) * c32 / qt) - \
                      (1 - Rhii) * ((1 + Rhii) * c41 / kz_iza[i] - (1 - Rhii) * c42 / qt) - \
                      (1 + Rhii) * ((1 + Rhii) * c51 / kz_iza[i] - (1 - Rhii) * c52 / qt)

    return Fvv, Fhh


cdef tuple compute_Fxxyxx(double complex[:] Rvi, double complex[:] Rhi, double complex[:] eps, double[:]
                          k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                          double[:] phi):

    cdef:
        double complex[:] Fvvupi, Fhhupi, Fvvups, Fhhups, Fvvdni, Fhhdni, Fvvdns, Fhhdns

    Fvvupi, Fhhupi = compute_Cm1(ud=1, Rvi=Rvi, Rhi=Rhi, eps=eps, k=k, kz_iza=kz_iza, kz_vza=kz_vza, iza=iza, vza=vza,
                                 raa=raa, phi=phi)
    Fvvups, Fhhups = compute_Cm2(ud=1, Rvi=Rvi, Rhi=Rhi, eps=eps, k=k, kz_iza=kz_iza, kz_vza=kz_vza, iza=iza, vza=vza,
                                 raa=raa, phi=phi)

    Fvvdni, Fhhdni = compute_Cm1(ud=-1, Rvi=Rvi, Rhi=Rhi, eps=eps, k=k, kz_iza=kz_iza, kz_vza=kz_vza, iza=iza, vza=vza,
                                 raa=raa, phi=phi)
    Fvvdns, Fhhdns = compute_Cm2(ud=-1, Rvi=Rvi, Rhi=Rhi, eps=eps, k=k, kz_iza=kz_iza, kz_vza=kz_vza, iza=iza, vza=vza,
                                 raa=raa, phi=phi)

    return Fvvupi, Fhhupi, Fvvups, Fhhups, Fvvdni, Fhhdni, Fvvdns, Fhhdns

# ----------------------------------------------------------------------------------------------------------------------
# Computation of IPP
# ----------------------------------------------------------------------------------------------------------------------
# Auxiliary Functions ----------------------------------------------------------------------------------------------
cdef double complex compute_Ixx(double qi, double qs, double kz_iza, double kz_vza, double sigma,
                       double complex fvv, double complex fhh, Py_ssize_t i, double complex Fxxupi,
                       double complex Fxxups, double complex Fxxdni, double complex Fxxdns):

    cdef:
        double complex Ixx
        double qi2, qs2, dkz, sigma2

    dkz = (kz_vza - kz_iza)
    qi2 = pow(qi, 2)
    qs2 = pow(qs, 2)
    sigma2 = pow(sigma, 2)

    Ixx = pow((kz_iza + kz_vza), i) * fvv * exp(-sigma2 * kz_iza * kz_vza) + 0.25 * \
          (Fxxupi * pow((kz_vza - qi), (i - 1)) * exp(-sigma2 * (qi2 - qi * dkz)) +
           Fxxdni * pow((kz_vza + qi), (i - 1)) * exp(-sigma2 * (qi2 + qi * dkz)) +
           Fxxups * pow((kz_iza + qs), (i - 1)) * exp(-sigma2 * (qs2 - qs * dkz)) +
           Fxxdns * pow((kz_iza - qs), (i - 1)) * exp(-sigma2 * (qs2 + qs * dkz)))

    return Ixx

# Computation of IPP -----------------------------------------------------------------------------------------------
cdef tuple compute_IPP(double[:] iza, double[:] vza, double[:] k, double[:] kz_iza, double[:] kz_vza, double[:] sigma,
                       double complex[:] fvv, double complex[:] fhh, int[:] Ts, double complex[:] Fvvupi,
                       double complex[:] Fhhupi, double complex[:] Fvvups, double complex[:] Fhhups,
                       double complex[:] Fvvdni, double complex[:] Fhhdni, double complex[:] Fvvdns,
                       double complex[:] Fhhdns):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t tmax = max(Ts.base)
        Py_ssize_t i, index

        # < Temporaly Variable Definitions > ------------
        double Gqi
        double qi, qs

        double sin_iza, sin_vza, sin_raa, sin_phi, ki, k2
        double cos_iza, cos_vza, cos_raa, cos_phi

        # < Definitions of C Parameter > ------------
        double c11, c12, c21, c31, c41, c42, c51
        double complex c22, c32, c52, Rvii, Rhii

        # < Definition of Fxx > ------------
        double complex[:, :] Ivv_view, Ihh_view

    Ivv = np.zeros((xmax, tmax), dtype=np.complex)
    Ihh = np.zeros((xmax, tmax), dtype=np.complex)

    Ivv_view = Ivv
    Ihh_view = Ihh

    for index in range(xmax):
        kz_izai = kz_iza[index]
        kz_vzai = kz_vza[index]

        fvvi = fvv[index]
        fhhi = fhh[index]

        sigmai = sigma[index]

        Fvvdnii = Fvvdni[index]
        Fvvupsi = Fvvups[index]
        Fvvdnsi = Fvvdns[index]
        Fvvupii = Fvvupi[index]

        Fhhdnii = Fhhdni[index]
        Fhhupsi = Fhhups[index]
        Fhhdnsi = Fhhdns[index]
        Fhhupii = Fhhupi[index]

        qi = k[index] * cos(iza[index])
        qs = k[index] * cos(vza[index])

        for i in range(1, Ts[index] + 1):
            Ivv_view[index, i-1] = compute_Ixx(qi, qs, kz_izai, kz_vzai, sigmai, fvvi, fhhi, i, Fvvupii, Fvvupsi, Fvvdnii,
                                             Fvvdnsi)

            Ihh_view[index, i-1] = compute_Ixx(qi, qs, kz_izai, kz_vzai, sigmai, fhhi, fhhi, i, Fhhupii, Fhhupsi, Fhhdnii,
                                              Fhhdnsi)

    return Ivv, Ihh

# ----------------------------------------------------------------------------------------------------------------------
# Computation of Sigma Nought
# ----------------------------------------------------------------------------------------------------------------------
cdef tuple compute_sigma_nought(int[:] Ts, double[:, :] Wn, double complex[:, :] Ivv, double complex[:, :] Ihh,
                                double[:] ShdwS, double[:] k, double[:] kz_iza, double[:] kz_vza, double[:] sigma):
    cdef:
        Py_ssize_t xmax = kz_iza.shape[0]
        Py_ssize_t i, index, i_index

        double sigmavv, sigmahh, a0, sigma2i, kz_xzai2, fact
        double[:] VV_view, HH_view

    VV = np.zeros(xmax, dtype=np.double)
    HH = np.zeros(xmax, dtype=np.double)

    VV_view = VV
    HH_view = HH

    sigmavv, sigmahh = 0.0, 0.0

    for i in range(xmax):
        sigma2i = pow(sigma[i], 2)
        kz_xzai2 = pow(kz_iza[i], 2) + pow(kz_vza[i], 2)

        for j in range(1, Ts[i] + 1):
            index = j - 1

            a0 = Wn[i, index] / factorial(j) * pow(sigma[i], (2 * j))

            sigmavv += pow(abs(Ivv[i, index]), 2) * a0
            sigmahh += pow(abs(Ihh[i, index]), 2) * a0

        VV_view[i] = sigmavv * ShdwS[i] * pow(k[i], 2) / 2 * np.exp(-sigma2i * kz_xzai2)
        HH_view[i] = sigmahh * ShdwS[i] * pow(k[i], 2) / 2 * np.exp(-sigma2i * kz_xzai2)

    return VV, HH

# ----------------------------------------------------------------------------------------------------------------------
# Shadowing Function
# ----------------------------------------------------------------------------------------------------------------------
cdef double[:] compute_ShdwS(double[:] iza, double[:] vza, double[:] raa, double[:] rss):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i

        double ct, cts, rslp, ctorslp, ctsorslp, shadf, shadfs

        double[:] ShdwS_view

    ShdwS = np.zeros((xmax), dtype=np.double)
    ShdwS_view = ShdwS

    for i in range(xmax):
        if iza[i] == vza[i] and np.allclose(raa[i], PI):
            ct = cos(iza[i]) / sin(iza[i])
            cts = cos(vza[i]) / sin(vza[i])
            rslp = rss[i]
            ctorslp = ct / sqrt(2) / rslp
            ctsorslp = cts / sqrt(2) / rslp
            shadf = 0.5 * (exp(-ctorslp ** 2) / sqrt(PI) / ctorslp - erf(ctorslp))
            shadfs = 0.5 * (exp(-ctsorslp ** 2) / sqrt(PI) / ctsorslp - erf(ctsorslp))

            ShdwS_view[i] = 1 / (1 + shadf + shadfs)
        else:
            ShdwS_view[i] = 1.0

    return ShdwS


# ----------------------------------------------------------------------------------------------------------------------
# Computation if I2EM
# ----------------------------------------------------------------------------------------------------------------------
cdef tuple compute_i2em(double[:] k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                        double[:] phi, double complex[:] eps, double[:] corrlength, double[:] sigma,
                        int[:] n, corrfunc):

    cdef:
        tuple Fxxyxx
        double complex[:] rt, Rvi, Rhi, Rv0, Rh0, Ft, Fvvupi, Fhhupi, Fvvups, Fhhups, Fvvdni, Fhhdni, Fvvdns, Fhhdns
        double complex[:] Rvt, Rht, fvv, fhh, RaV, RaH
        double complex[:, :] Ivv, Ihh
        double[:] wvnb, ShdwS, VV, HH, Tf
        int[:] Ts

    # Subsection3 --------------------------------------------------------------------------------------------------
    Wn, rss = compute_Wn_rss(corrfunc=corrfunc, iza=iza, vza=vza, raa=raa, phi=phi, k=k, sigma=sigma,
                             corrlength=corrlength, n=n)

    Ts = compute_TS(iza=iza, vza=vza, sigma=sigma, k=k)

    # Reflection Coefficients --------------------------------------------------------------------------------------
    rt = compute_rt(iza=iza, epsr=eps.base.real, epsi=eps.base.imag)
    Rvi, Rhi = compute_Rxi(iza=iza, eps=eps, rt=rt)
    wvnb = compute_wvnb(iza=iza, vza=vza, raa=raa, phi=phi, k=k)

    # Shadowing Function -------------------------------------------------------------------------------------------
    ShdwS = compute_ShdwS(iza=iza, vza=vza, raa=raa, rss=rss)

    # R-Transition -------------------------------------------------------------------------------------------------
    Rv0, Rh0 = compute_Rx0(eps)
    Ft = compute_Ft(iza=iza, vza=vza, eps=eps)
    Tf = compute_Tf(iza=iza, k=k, sigma=sigma, Rv0=Rv0, Ft=Ft, Wn=Wn, Ts=Ts)

    # RaX Integration ----------------------------------------------------------------------------------------------
    RaV, RaH = Rax_integration(iza=iza, sigma=sigma, corrlength=corrlength, eps=eps)

    # Bistatic Coefficients ----------------------------------------------------------------------------------------
    Rvt, Rht = compute_Rxt(iza=iza, vza=vza, raa=raa, sigma=sigma, corrlength=corrlength, eps=eps, Tf=Tf)
    fvv, fhh = compute_fxx(Rvt=Rvt, Rht=Rht, iza=iza, vza=vza, raa=raa)

    Fxxyxx = compute_Fxxyxx(Rvi=Rvi, Rhi=Rhi, eps=eps, k=k, kz_iza=kz_iza, kz_vza=kz_vza, iza=iza, vza=vza,
                            raa=raa, phi=phi)

    Fvvupi, Fhhupi, Fvvups, Fhhups, Fvvdni, Fhhdni, Fvvdns, Fhhdns = Fxxyxx

    Ivv, Ihh = compute_IPP(iza=iza, vza=vza, k=k, kz_iza=kz_iza, kz_vza=kz_vza, sigma=sigma, fvv=fvv, fhh=fhh, Ts=Ts,
                           Fvvupi=Fvvupi, Fhhupi=Fhhupi, Fvvups=Fvvups, Fhhups=Fhhups, Fvvdni=Fvvdni,
                           Fhhdni=Fhhdni, Fvvdns=Fvvdns, Fhhdns=Fhhdns)

    # Sigma Nought -------------------------------------------------------------------------------------------------
    VV, HH = compute_sigma_nought(Ts=Ts, Wn=Wn, Ivv=Ivv, Ihh=Ihh,
                                  ShdwS=ShdwS, k=k, kz_iza=kz_iza, kz_vza=kz_vza, sigma=sigma)


    # __PAR__ = ['wn', 'rss', 'Ts', 'rt', 'Rvi', 'Rhi', 'wvnb', 'ShdwS', 'Rv0', 'Rh0', 'Ft', 'Tf', 'RaV', 'RaH', 'Rvt', 'Rht', 'fvv', 'fhh', 'Fvvupi', 'Fhhupi', 'Fvvups', 'Fhhups', 'Fvvdni', 'Fhhdni', 'Fvvdns', 'Fhhdns', 'Ivv', 'Ihh', 'VV', 'HH']
    #
    # __VAL__ = [Wn, rss, Ts.base, rt.base, Rvi.base, Rhi.base, wvnb.base, ShdwS.base, Rv0.base, Rh0.base, Ft.base, Tf.base, RaV.base, RaH.base, Rvt.base, Rht.base, fvv.base, fhh.base, Fvvupi.base, Fhhupi.base, Fvvups.base, Fhhups.base, Fvvdni.base, Fhhdni.base, Fvvdns.base, Fhhdns.base, Ivv.base, Ihh.base, VV.base, HH.base]
    #
    # for i, item in enumerate(__PAR__):
    #     sys.stdout.write(item + ' = {0}{1}'.format(str(__VAL__[i]), str('\n')))

    return VV, HH

def i2em_wrapper(double[:] k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                 double[:] phi, double complex[:] eps, double[:] corrlength, double[:] sigma, corrfunc, int[:] n):

    return compute_i2em(k=k, kz_iza=kz_iza, kz_vza=kz_vza, iza=iza, vza=vza, raa=raa, phi=phi, eps=eps,
                        corrlength=corrlength, sigma=sigma, corrfunc=corrfunc, n=n)

# Compute I2EM Emissivity ##############################################################################################
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
        svv[n, :] = (de * (vv + hv) * sin(x) * (1 / cos(iza))) / (4 * PI)

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
        (2 * (3 + 4) * sin(5) * 1 / cos(6)) / (PI * 4)
        shh[n, :] = (de * (hh + vh) * sin(x) * (1 / cos(iza))) / (4 * PI)

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

    cdef float refv = dblquad(emsv_integralfunc, 0, PI / 2, lambda x: 0, lambda x: PI,
                              args=(iza, eps, rv, rh, k, kl, ks, sq, corrfunc_ems, corrlength, 1))[0]

    cdef float refh = dblquad(emsv_integralfunc, 0, PI / 2, lambda x: 0, lambda x: PI,
                              args=(iza, eps, rv, rh, k, kl, ks, sq, corrfunc_ems, corrlength, 2))[0]

    cdef float VV = 1 - refv - np.exp(-ks ** 2 * np.cos(iza) * np.cos(iza)) * (
        abs(rv)) ** 2
    cdef float HH = 1 - refh - np.exp(-ks ** 2 * np.cos(iza) * np.cos(iza)) * (
        abs(rh)) ** 2

    return VV, HH

def calc_iem_ems_wrapper(float iza, float k, float sigma, float corrlength, double complex eps, int corrfunc_ems):
    return calc_iem_ems(iza, k, sigma, corrlength, eps, corrfunc_ems)
