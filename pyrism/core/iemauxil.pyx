# cython: cdivision=True
from __future__ import division

cimport numpy as np
import numpy as np
import cmath
from scipy.integrate import dblquad
import cython
from scipy.special import erf
from libc.math cimport sin, cos, pow, sqrt, exp

DTYPE = np.float

ctypedef np.float_t DTYPE_t

cdef:
    double PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164

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

cdef tuple compute_Rxi(double[:] iza, complex[:] eps, double complex[:] rt):
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
        mui = complex(cos(iza[i]), 0)
        Rvi_view[i] = (eps[i] * mui - rt[i]) / (eps[i] * mui + rt[i])
        Rhi_view[i] = (mui - rt[i]) / (mui + rt[i])

    return Rvi, Rhi

# ----------------------------------------------------------------------------------------------------------------------
# Computation of RT, WVNB, Ts and Ft
# ----------------------------------------------------------------------------------------------------------------------
cdef double complex[:] compute_rt(double[:] iza, double[:] epsr, double[:] epsi):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double complex[:] rt_view
        double temp

    rt = np.zeros_like(iza, dtype=np.complex)
    rt_view = rt

    for i in range(xmax):
        temp = sqrt(epsr[i] - pow(sin(iza[i]), 2))
        rt[i] = complex(temp, epsi[i])

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
        error = 1.0e8

        while error >= 1.0e-3 and TS_temp <= 150:
            TS_temp += 1
            error = ((k[i] * sigma[i]) ** 2 * pow((mui + muv), 2) ** TS_temp) / factorial(TS_temp)

        TS_view[i] = TS_temp

    return TS

def TS_wrapper(double[:] iza, double[:] vza, double[:] sigma, double[:] k):
    return compute_TS(iza, vza, sigma, k)

cdef double complex[:] compute_Ft(double[:] iza, double[:] vza, double complex[:] eps):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double complex[:] Rv0, Ft_view
        double sin_iza, sin_vza, cos_iza, Rv02

    Rv0 = compute_Rx0(eps)

    Ft = np.zeros_like(iza, dtype=np.double)
    Ft_view = Ft

    for i in range(xmax):
        sin_iza = sin(iza[i])
        sin_vza = sin(vza[i])
        cos_iza = cos(iza[i])
        Rv02 = cmath.pow(Rv0[i], 2)

        Ft_view[i] = 8 * Rv02 * sin_vza * (cos_iza + cmath.sqrt(eps[i] - pow(sin_iza, 2))) / (cos_iza * cmath.sqrt(eps[i] - pow(sin_iza, 2)))

    return Ft

cdef double[:] compute_Tf(double[:] iza, double[:] k, double[:] sigma, double complex[:] Rv0, double complex[:] Ft,
                          int[:] Wn):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t x, i
        double a0, a1, b1, temp, St, St0
        double[:] Tf_view

    Tf = np.zeros_like(iza, dtype=np.double)
    Tf_view = Tf

    a1, b1 = 0.0, 0.0

    for i in range(xmax):
        x += 1
        cos_iza = cos(iza[i])
        a0 = pow(k[i] * sigma[i] * cos_iza, 2*x) / factorial(x)
        a1 += a0 * Wn[i]
        temp = abs(Ft[i] / 2 + pow(2, x) * Rv0[i] / cos_iza * exp(- pow((k[i] * sigma[i]) * cos_iza, 2)))
        b1 += a0 * a0 * pow(temp, 2) * Wn[i]

        St = 0.25 * (abs(Ft[i]) ** 2) * a1 / b1
        St0 = 1 / (abs(1 + 8 * Rv0[i] / (cos_iza * Ft[i]))) ** 2

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

    A = cos_iza + Zx * sin_iza
    B = (1 + pow(Zx, 2) + pow(Zy, 2)) * eps
    CC = pow(sin_iza, 2) - 2 * Zx * sin_iza * cos_iza + pow(Zx, 2) * pow(cos_iza, 2) + pow(Zy, 2)

    return A, B, CC

# Callable Integration Functions -----------------------------------------------------------------------------------
cdef double RaV_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps):
    cdef:
        double A, CC, pd
        double complex B, Rv, Rav

    A, B, CC = compute_ABCC(Zy, Zx, iza, eps)
    pd = exp(-pow(Zx, 2) / (2 * pow(sigx, 2)) - pow(Zy, 2) / (2 * pow(sigy, 2)))

    Rv = (eps * A - cmath.sqrt(B - CC)) / (eps * A + cmath.sqrt(B - CC))
    Rav = Rv * pd

    return Rav.real

cdef double RaH_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps):
    cdef:
        double A, CC, pd
        double complex B, Rh, Rah

    A, B, CC = compute_ABCC(Zy, Zx, iza, eps)
    pd = exp(-pow(Zx, 2) / (2 * pow(sigx, 2)) - pow(Zy, 2) / (2 * pow(sigy, 2)))

    Rh = (A - cmath.sqrt(B - CC)) / (A + cmath.sqrt(B - CC))
    RaH = Rh * pd

    return RaH.real

# Integration with SciPy -------------------------------------------------------------------------------------------
cdef tuple Rax_integration(double[:] iza, double[:] sigma, double[:] corrlength, double complex[:] eps):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double bound, ravv, rahh
        double[:] Rav_view, Rah_view
        double sigy, sigx

    Rav = np.zeros_like(iza, dtype=np.double)
    Rah = np.zeros_like(iza, dtype=np.double)
    Rav_view = Rav
    Rah_view = Rah

    for i in range(xmax):
        sigx = 1.1 * sigma[i] / corrlength[i]
        sigy = sigx

        bound = 3 * sigx

        ravv = dblquad(RaV_integration_ifunc, -bound, bound, lambda x: -bound, lambda x: bound,
                   args=(iza[i], sigx, sigy, eps[i]))[0]

        rahh = dblquad(RaH_integration_ifunc, -bound, bound, lambda x: -bound, lambda x: bound,
                   args=(iza[i], sigx, sigy, eps[i]))[0]

        Rav_view[i] = ravv / (2 * PI * sigx * sigy)
        Rah_view[i] = rahh / (2 * PI * sigx * sigy)

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
        double[:] RaV, RaH
        double complex[:] Rvt_view, Rht_view, rt, Rv0, Rh0, Rvi, Rhi

    rt = compute_rt(iza, eps.real, eps.imag)

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
            Rvt_view[i] = complex(RaV[i], 0)
            Rht_view[i] = complex(RaH[i], 0)

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
        fhh_view[i] = 2 * Rht[i] * temp

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
        k2 = pow(k[i], 2)

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


        qt = k * cmath.sqrt(eps[i] - pow(sin_iza, 2))
    
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
        k2 = pow(k[i], 2)

        Gqs = ud * kz_vza[i]
        Gqts = ud * ki * cmath.sqrt(eps[i] - pow(sin_iza, 2))

        # < Matrix Elements > ------------
        c11 = ki * cos_raa * (kz_iza[i] + Gqs)
        c12 = c11

        c21 = Gqs * (cos_raa * (cos_iza * (ki * cos_iza + Gqs) -
                                ki * sin_iza * (sin_vza * cos_raa - sin_iza * cos_phi)) -
                     ki * sin_iza * sin_vza * sin_phi ** 2)
        c22 = Gqts * (cos_raa * (cos_iza * (kz_iza + Gqs) -
                                 ki * sin_iza * (sin_vza * cos_raa - sin_iza * cos_phi)) -
                      ki * sin_iza * sin_vza * sin_phi ** 2)

        c31 = ki * sin_vza * (ki * cos_iza * (sin_vza * cos_raa - sin_iza * cos_phi) +
                              sin_iza * (kz_iza + Gqs))
        c32 = c31

        c41 = ki * cos_vza * (cos_raa * (cos_iza * (kz_iza + Gqs) - ki * sin_iza *
                                         (sin_vza * cos_raa - sin_iza * cos_phi)) -
                              ki * sin_iza * sin_vza * sin_phi ** 2)
        c42 = c41

        c51 = -cos_vza * (k2 * sin_vza * (sin_vza * cos_raa - sin_iza * cos_phi) +
                          Gqs * cos_raa * (kz_iza + Gqs))
        c52 = -cos_vza * (k2 * sin_vza * (sin_vza * cos_raa - sin_iza * cos_phi) +
                          Gqts * cos_raa * (kz_iza + Gqs))


        qt = k * cmath.sqrt(eps[i] - pow(sin_iza, 2))

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
        Py_ssize_t tmax = Ts.shape[0]
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

        for i in range(1, tmax+1):

            Ivv_view[index, i] = compute_Ixx(qi, qs, kz_izai, kz_vzai, sigmai, fvvi, fhhi, i, Fvvupii, Fvvupsi, Fvvdnii,
                                             Fvvdnsi)

            Ihh_view[index, i] = compute_Ixx(qi, qs, kz_izai, kz_vzai, sigmai, fhhi, fhhi, i, Fhhupii, Fhhupsi, Fhhdnii,
                                              Fhhdnsi)

    return Ivv, Ihh

# ----------------------------------------------------------------------------------------------------------------------
# Computation of Sigma Nought
# ----------------------------------------------------------------------------------------------------------------------
cdef tuple compute_sigma_nought(int[:] Ts, int[:] Wn, double complex[:, :] Ivv, double complex[:, :] Ihh,
                                double[:] ShdwS, double[:] k, double[:] kz_iza, double[:] kz_vza, double[:] sigma):
    cdef:
        Py_ssize_t xmax = kz_iza.shape[0]
        Py_ssize_t tmax = Ts.shape[0]
        Py_ssize_t i, index

        double sigmavv, sigmahh, a0, sigma2i, kz_xzai2
        double[:] VV_view, HH_view

    VV = np.zeros(xmax, dtype=np.double)
    HH = np.zeros(xmax, dtype=np.double)

    VV_view = VV
    HH_view = HH

    sigmavv, sigmahh = 0.0, 0.0

    for index in range(xmax):
        sigma2i = pow(sigma[index], 2)
        kz_xzai2 = pow(kz_iza[index], 2) + pow(kz_vza[index], 2)

        for i in range(1, tmax + 1):

            a0 = Wn[index] / factorial(i) * pow(sigma[index], (2 * i))

            sigmavv += pow(abs(Ivv[index, i - 1]), 2) * a0
            sigmahh += pow(abs(Ihh[index, i - 1]), 2) * a0

        VV_view[index] = sigmavv * ShdwS[index] * pow(k[index], 2) / 2 * np.exp(-sigma2i * kz_xzai2)
        HH_view[index] = sigmahh * ShdwS[index] * pow(k[index], 2) / 2 * np.exp(-sigma2i * kz_xzai2)

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
                        double[:] phi, double complex[:] eps, double[:] corrlength, double[:] sigma, int[:] Wn,
                        double[:] rss):

    cdef:
        tuple Fxxyxx
        double complex[:] rt, Rvi, Rhi, Rv0, Rh0, Ft, Fvvupi, Fhhupi, Fvvups, Fhhups, Fvvdni, Fhhdni, Fvvdns, Fhhdns
        double complex[:] RaV, RaH, Rvt, Rht, fvv, fhh
        double complex[:, :] Ivv, Ihh
        double[:] wvnb, ShdwS, VV, HH, Tf
        int[:] Ts
        
    # Reflection Coefficients --------------------------------------------------------------------------------------
    rt = compute_rt(iza=iza, epsr=eps.real, epsi=eps.imag)
    Rvi, Rhi = compute_Rxi(iza=iza, eps=eps, rt=rt)
    wvnb = compute_wvnb(iza=iza, vza=vza, raa=raa, phi=phi, k=k)
    Ts = compute_TS(iza=iza, vza=vza, sigma=sigma, k=k)

    # Shadowing Function -------------------------------------------------------------------------------------------
    ShdwS = compute_ShdwS(iza=iza, vza=vza, raa=raa, rss=rss)

    # R-Transition -------------------------------------------------------------------------------------------------
    Rv0, Rh0 = compute_Rx0(eps)
    Ft = compute_Ft(iza=iza, vza=vza, eps=eps)
    Tf = compute_Tf(iza=iza, k=k, sigma=sigma, Rv0=Rv0, Ft=Ft, Wn=Wn)

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

    return VV, HH

def i2em_wrapper(double[:] k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                 double[:] phi, double complex[:] eps, double[:] corrlength, double[:] sigma, int[:] Wn,
                 double[:] rss):
    
    return compute_i2em(k=k, kz_iza=kz_iza, kz_vza=kz_vza, iza=iza, vza=vza, raa=raa, phi=phi, eps=eps,
                        corrlength=corrlength, sigma=sigma, Wn=Wn, rss=rss)


# Subsection1 ----------------------------------------------------------------------------------------------------------

# def calc_i2em_auxil(float k, float kz_iza, float kz_vza, double complex iza, double complex vza, double complex raa,
#                     double complex phi, double complex eps, double complex corrlength, double complex sigma,
#                     corrfunc, int n):
#     cdef double complex rt, Rvi, Rhi, wvnb, Tf, Rv0, Rh0, RaV, RaH
#     cdef int Ts
#     cdef list Ivv, Ihh
#     cdef np.ndarray Ivva, Ihha
#
#     cdef double complex siza = np.sin(iza + 0.01)
#     cdef double complex ciza = np.cos(iza + 0.01)
#     cdef double complex svza = np.sin(vza)
#     cdef double complex cvza = np.cos(vza)
#     cdef double complex sraa = np.sin(raa)
#     cdef double complex craa = np.cos(raa)
#     cdef double complex cphi = np.cos(phi)
#
#     rt, Rvi, Rhi, wvnb, Ts = reflection_coefficients(k, iza, vza, raa, phi, eps, sigma)
#
#     Wn, rss = corrfunc(sigma.real, corrlength.real, wvnb.real, Ts, n=n)
#     ShdwS = shadowing_function(iza.real, vza.real, raa.real, rss)
#
#     Tf, Rv0, Rh0 = r_transition(k, iza, vza, sigma, eps, Wn, Ts)
#     RaV, RaH = Ra_integration(iza, sigma, corrlength, eps)
#     fvv, fhh, Rvt, Rht = biStatic_coefficient(iza.real, vza.real, raa.real, Rvi.real, Rv0.real, Rhi.real, Rh0.real,
#                                               RaV.real, RaH.real,
#                                               Tf.real)
#
#     Ivv, Ihh = Ipp(siza, ciza, svza, cvza, sraa, craa, cphi, Rvi, Rhi, eps, k,
#                    kz_iza, kz_vza, fvv, fhh, sigma, Ts)
#
#     Ivva, Ihha = np.asarray(Ivv, dtype=np.complex), np.asarray(Ihh, dtype=np.complex)
#
#     VV, HH = sigma_nought(Ts, Wn, Ivva, Ihha, ShdwS, k, kz_iza, kz_vza, sigma.real)
#
#     return VV, HH


# def reflection_coefficients(float k, double complex iza, double complex vza, double complex raa, double complex phi,
#                             double complex eps,
#                             double complex sigma):
#     cdef DTYPE_t error
#
#     cdef double complex rt = np.sqrt(eps - pow(np.sin(iza + 0.01), 2))
#     cdef double complex Rvi = (eps * np.cos(iza + 0.01) - rt) / (eps * np.cos(iza + 0.01) + rt)
#     cdef double complex Rhi = (np.cos(iza + 0.01) - rt) / (np.cos(iza + 0.01) + rt)
#     cdef double complex wvnb = k * np.sqrt(
#         pow((np.sin(vza) * np.cos(raa) - np.sin(iza + 0.01) * np.cos(phi)), 2) + pow((
#                 np.sin(vza) * np.sin(raa) - np.sin(iza + 0.01) * np.sin(phi)), 2))
#
#     cdef int Ts = 1
#
#     cdef float merror = 1.0e8
#     while merror >= 1.0e-3 and Ts <= 150:
#         Ts += 1
#         error = ((k * sigma) ** 2 * (np.cos(iza + 0.01) + np.cos(vza)) ** 2) ** Ts / factorial(Ts)
#         merror = np.mean(error)
#
#     return rt, Rvi, Rhi, wvnb, Ts

# def r_transition(float k, double complex iza, double complex vza, double complex sigma, double complex eps,
#                  np.ndarray Wn, int Ts):
#     cdef int i
#     cdef double complex a0
#
#     cdef double complex Rv0 = (cmath.sqrt(eps) - 1) / (cmath.sqrt(eps) + 1) # done
#     cdef double complex Rh0 = -Rv0 # done
#
#     cdef double complex Ft = 8 * Rv0 ** 2 * cmath.sin(vza) * (np.cos(iza + 0.01) + cmath.sqrt(eps - cmath.sin(iza + 0.01) ** 2)) / (
#                                      np.cos(iza + 0.01) * cmath.sqrt(eps - cmath.sin(iza + 0.01) ** 2))
#     cdef double complex a1 = 0.0
#     cdef double complex b1 = 0.0
#
#     for i in range(Ts):
#         i += 1
#         a0 = ((k * sigma) * np.cos(iza + 0.01)) ** (2 * i) / factorial(i)
#         a1 = a1 + a0 * Wn[i - 1]
#         b1 = b1 + a0 * (abs(Ft / 2 + 2 ** (i + 1) * Rv0 / np.cos(iza + 0.01) * np.exp(- ((k * sigma) * np.cos(iza + 0.01)) ** 2))) ** 2 * Wn[i - 1]
#
#     cdef double complex St = 0.25 * (abs(Ft) ** 2) * a1 / b1
#     cdef double complex St0 = 1 / (abs(1 + 8 * Rv0 / (np.cos(iza + 0.01) * Ft))) ** 2
#     cdef double complex Tf = 1 - St / St0
#
#     return Tf, Rv0, Rh0

# def RaV_integration_function(DTYPE_t Zy, DTYPE_t Zx, double complex iza, double complex sigx, double complex sigy,
#                              double complex eps):
#     cdef double complex A = np.cos(iza + 0.01) + Zx * cmath.sin(iza + 0.01)
#     cdef double complex B = (1 + Zx ** 2 + Zy ** 2) * eps
#
#     cdef double complex CC = cmath.sin(iza + 0.01) ** 2 - 2 * Zx * cmath.sin(iza + 0.01) * np.cos(
#         iza + 0.01) + Zx ** 2 * np.cos(
#         iza + 0.01) ** 2 + Zy ** 2
#
#     cdef double complex Rv = (eps * A - np.sqrt(B - CC)) / (eps * A + np.sqrt(B - CC))
#
#     cdef double complex pd = np.exp(-Zx ** 2 / (2 * sigx ** 2) - Zy ** 2 / (2 * sigy ** 2))
#
#     cdef double complex Rav = Rv * pd
#
#     return Rav.real

# def RaH_integration_function(DTYPE_t Zy, DTYPE_t Zx, double complex iza, double complex sigx, double complex sigy,
#                              double complex eps):
#     cdef double complex A = np.cos(iza + 0.01) + Zx * cmath.sin(iza + 0.01)
#     cdef double complex B = eps * (1 + Zx ** 2 + Zy ** 2)
#
#     cdef double complex CC = cmath.sin(iza + 0.01) ** 2 - 2 * Zx * cmath.sin(iza + 0.01) * np.cos(
#         iza + 0.01) + Zx ** 2 * np.cos(
#         iza + 0.01) ** 2 + Zy ** 2
#
#     cdef double complex Rh = (A - np.sqrt(B - CC)) / (A + np.sqrt(B - CC))
#
#     cdef double complex pd = np.exp(-Zx ** 2 / (2 * sigx ** 2) - Zy ** 2.0 / (2 * sigy ** 2))
#
#     cdef double complex RaH = Rh * pd
#
#     return RaH.real

# def RaV_integration(double complex iza, double complex sigx, double complex sigy, double complex eps):
#     cdef double complex ravv, Rav
#
#     cdef float bound = 3 * sigx.real
#
#     ravv = dblquad(RaV_integration_function, -bound, bound, lambda x: -bound, lambda x: bound,
#                    args=(iza, sigx, sigy, eps))[0]
#     temp = np.asarray(ravv) / (2 * PI * sigx * sigy)
#
#     Rav = np.asarray(np.asarray(ravv) / (2 * PI * sigx * sigy))
#
#     return Rav


# def RaH_integration(double complex iza, double complex sigx, double complex sigy, double complex eps):
#     cdef double complex rahh, RaH
#
#     cdef float bound = 3 * sigx.real
#
#     rahh = dblquad(RaH_integration_function, -bound, bound, lambda x: -bound, lambda x: bound,
#                    args=(iza, sigx, sigy, eps))[0]
#
#     RaH = np.asarray(np.asarray(rahh) / (2 * PI * sigx * sigy))
#
#     return RaH


# def Ra_integration(double complex iza, double complex sigma, double complex corrlength, double complex eps):
#     cdef double complex sigx = 1.1 * sigma / corrlength
#     cdef double complex sigy = sigx
#
#     cdef double complex RaV = RaV_integration(iza, sigx, sigy, eps)
#     cdef double complex RaH = RaH_integration(iza, sigx, sigy, eps)
#
#     return RaV, RaH

# def biStatic_coefficient(DTYPE_t iza, DTYPE_t vza, DTYPE_t raa, DTYPE_t Rvi, DTYPE_t Rv0, DTYPE_t Rhi, DTYPE_t Rh0,
#                          DTYPE_t RaV, DTYPE_t RaH, DTYPE_t Tf):
#     cdef DTYPE_t Rvt, Rht, fvv, fhh
#
#     if np.array_equal(vza, iza) and (np.allclose(np.all(raa), PI)):
#         Rvt = Rvi + (Rv0 - Rvi) * Tf
#         Rht = Rhi + (Rh0 - Rhi) * Tf
#
#     else:
#         Rvt = RaV
#         Rht = RaH
#
#     fvv = 2 * Rvt * (
#             sin(iza + 0.01) * sin(vza) - (1 + np.cos(iza + 0.01) * np.cos(vza)) * np.cos(raa)) / (
#                   np.cos(iza + 0.01) + np.cos(vza))
#
#     fhh = -2 * Rht * (
#             sin(iza + 0.01) * sin(vza) - (1 + np.cos(iza + 0.01) * np.cos(vza)) * np.cos(raa)) / (
#                   np.cos(iza + 0.01) + np.cos(vza))
#
#     return fvv, fhh, Rvt, Rht


# def calc_iem_ems_wrapper(float iza, float k, float sigma, float corrlength, double complex eps, int corrfunc_ems):
#     return calc_iem_ems(iza, k, sigma, corrlength, eps, corrfunc_ems)

# def Fppupdn_calc(double complex ud, int method, double complex Rvi, double complex Rhi, double complex er,
#                  float k, float kz, float ksz, double complex s, double complex cs, ss, css,
#                  double complex cf, double complex cfs, double complex sfs):
#
#     cdef double complex Gqi, Gqti, qi, c11, c21, c31, c41, c51, c12, c22, c32, c42, c52, q, qt, vv, hh
#     if method == 1:
#         Gqi = ud * kz
#         Gqti = ud * k * np.sqrt(er - s ** 2)
#         qi = ud * kz
#
#         c11 = k * cfs * (ksz - qi)
#         c21 = cs * (cfs * (
#                 k ** 2 * s * cf * (ss * cfs - s * cf) + Gqi * (k * css - qi)) + k ** 2 * cf * s * ss * sfs ** 2)
#         c31 = k * s * (s * cf * cfs * (k * css - qi) - Gqi * (cfs * (ss * cfs - s * cf) + ss * sfs ** 2))
#         c41 = k * cs * (cfs * css * (k * css - qi) + k * ss * (ss * cfs - s * cf))
#         c51 = Gqi * (cfs * css * (qi - k * css) - k * ss * (ss * cfs - s * cf))
#
#         c12 = k * cfs * (ksz - qi)
#         c22 = cs * (cfs * (
#                 k ** 2 * s * cf * (ss * cfs - s * cf) + Gqti * (k * css - qi)) + k ** 2 * cf * s * ss * sfs ** 2)
#         c32 = k * s * (s * cf * cfs * (k * css - qi) - Gqti * (cfs * (ss * cfs - s * cf) - ss * sfs ** 2))
#         c42 = k * cs * (cfs * css * (k * css - qi) + k * ss * (ss * cfs - s * cf))
#         c52 = Gqti * (cfs * css * (qi - k * css) - k * ss * (ss * cfs - s * cf))
#
#     if method == 2:
#         Gqs = ud * ksz
#         Gqts = ud * k * np.sqrt(er - ss ** 2)
#         qs = ud * ksz
#
#         c11 = k * cfs * (kz + qs)
#         c21 = Gqs * (cfs * (cs * (k * cs + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs ** 2)
#         c31 = k * ss * (k * cs * (ss * cfs - s * cf) + s * (kz + qs))
#         c41 = k * css * (cfs * (cs * (kz + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs ** 2)
#         c51 = -css * (k ** 2 * ss * (ss * cfs - s * cf) + Gqs * cfs * (kz + qs))
#
#         c12 = k * cfs * (kz + qs)
#         c22 = Gqts * (cfs * (cs * (kz + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs ** 2)
#         c32 = k * ss * (k * cs * (ss * cfs - s * cf) + s * (kz + qs))
#         c42 = k * css * (cfs * (cs * (kz + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs ** 2)
#         c52 = -css * (k ** 2 * ss * (ss * cfs - s * cf) + Gqts * cfs * (kz + qs))
#
#     q = kz
#     qt = k * np.sqrt(er - s ** 2)
#
#     vv = (1 + Rvi) * (-(1 - Rvi) * c11 / q + (1 + Rvi) * c12 / qt) + \
#          (1 - Rvi) * ((1 - Rvi) * c21 / q - (1 + Rvi) * c22 / qt) + \
#          (1 + Rvi) * ((1 - Rvi) * c31 / q - (1 + Rvi) * c32 / er / qt) + \
#          (1 - Rvi) * ((1 + Rvi) * c41 / q - er * (1 - Rvi) * c42 / qt) + \
#          (1 + Rvi) * ((1 + Rvi) * c51 / q - (1 - Rvi) * c52 / qt)
#
#     hh = (1 + Rhi) * ((1 - Rhi) * c11 / q - er * (1 + Rhi) * c12 / qt) - \
#          (1 - Rhi) * ((1 - Rhi) * c21 / q - (1 + Rhi) * c22 / qt) - \
#          (1 + Rhi) * ((1 - Rhi) * c31 / q - (1 + Rhi) * c32 / qt) - \
#          (1 - Rhi) * ((1 + Rhi) * c41 / q - (1 - Rhi) * c42 / qt) - \
#          (1 + Rhi) * ((1 + Rhi) * c51 / q - (1 - Rhi) * c52 / qt)
#
#     return vv, hh

# def Ipp(double complex siza, double complex ciza, double complex svza, double complex cvza,
#         double complex sraa, double complex craa, double complex cphi, double complex Rvi,
#         double complex Rhi, double complex eps, float k, float kz_iza, float kz_vza,
#         double complex fvv, double complex fhh, double complex sigma, int Ts):
#     cdef int i
#     Fvvupi, Fhhupi = Fppupdn_calc(+1, 1,
#                                   Rvi,
#                                   Rhi,
#                                   eps,
#                                   k,
#                                   kz_iza,
#                                   kz_vza,
#                                   siza,
#                                   ciza,
#                                   svza,
#                                   cvza,
#                                   cphi,
#                                   craa,
#                                   sraa)
#
#     Fvvups, Fhhups = Fppupdn_calc(+1, 2,
#                                   Rvi,
#                                   Rhi,
#                                   eps,
#                                   k,
#                                   kz_iza,
#                                   kz_vza,
#                                   siza,
#                                   ciza,
#                                   svza,
#                                   cvza,
#                                   cphi,
#                                   craa,
#                                   sraa)
#
#     Fvvdni, Fhhdni = Fppupdn_calc(-1, 1,
#                                   Rvi,
#                                   Rhi,
#                                   eps,
#                                   k,
#                                   kz_iza,
#                                   kz_vza,
#                                   siza,
#                                   ciza,
#                                   svza,
#                                   cvza,
#                                   cphi,
#                                   craa,
#                                   sraa)
#
#     Fvvdns, Fhhdns = Fppupdn_calc(-1, 2,
#                                   Rvi,
#                                   Rhi,
#                                   eps,
#                                   k,
#                                   kz_iza,
#                                   kz_vza,
#                                   siza,
#                                   ciza,
#                                   svza,
#                                   cvza,
#                                   cphi,
#                                   craa,
#                                   sraa)
#
#     cdef double complex qi = k * ciza
#     cdef double complex qs = k * cvza
#
#     ivv = []
#     ihh = []
#     for i in range(Ts):
#         i += 1
#         Ivv = (kz_iza + kz_vza) ** i * fvv * np.exp(
#             -sigma ** 2 * kz_iza * kz_vza) + \
#               0.25 * (Fvvupi * (kz_vza - qi) ** (i - 1) * np.exp(
#             -sigma ** 2 * (pow(qi, 2) - qi * (kz_vza - kz_iza))) + Fvvdni * (
#                               kz_vza + qi) ** (i - 1) * np.exp(
#             -sigma ** 2 * (pow(qi, 2) + qi * (kz_vza - kz_iza))) + Fvvups * (
#                               kz_iza + qs) ** (i - 1) * np.exp(
#             -sigma ** 2 * (qs ** 2 - qs * (kz_vza - kz_iza))) + Fvvdns * (
#                               kz_iza - qs) ** (i - 1) * np.exp(
#             -sigma ** 2 * (qs ** 2 + qs * (kz_vza - kz_iza))))
#
#         Ihh = (kz_iza + kz_vza) ** i * fhh * np.exp(
#             -sigma ** 2 * kz_iza * kz_vza) + \
#               0.25 * (Fhhupi * (kz_vza - qi) ** (i - 1) * np.exp(
#             -sigma ** 2 * (pow(qi, 2) - qi * (kz_vza - kz_iza))) +
#                       Fhhdni * pow((kz_vza + qi), (i - 1)) * np.exp(
#                     -sigma ** 2 * (pow(qi, 2) + qi * (kz_vza - kz_iza))) +
#                       Fhhups * pow((kz_iza + qs), (i - 1)) * np.exp(
#                     -sigma ** 2 * (qs ** 2 - qs * (kz_vza - kz_iza))) +
#                       Fhhdns * (kz_iza - qs) ** (i - 1) * np.exp(
#                     -sigma ** 2 * (qs ** 2 + qs * (kz_vza - kz_iza))))
#
#         ivv.append(Ivv)
#         ihh.append(Ihh)
#     Ivv = ivv
#     Ihh = ihh
#
#     return Ivv, Ihh

# def sigma_nought(int Ts, np.ndarray Wn, np.ndarray Ivv, np.ndarray Ihh, double complex ShdwS, float k,
#                  float kz_iza, float kz_vza, double complex sigma):
#     cdef float sigmavv = 0
#     cdef float sigmahh = 0
#     for i in range(Ts):
#         i += 1
#         a0 = Wn[i - 1] / factorial(i) * sigma ** (2 * i)
#
#         sigmavv = sigmavv + np.abs(Ivv[i - 1]) ** 2 * a0
#         sigmahh = sigmahh + np.abs(Ihh[i - 1]) ** 2 * a0
#
#     cdef float VV = sigmavv * ShdwS * k ** 2 / 2 * np.exp(
#         -sigma ** 2 * (kz_iza ** 2 + kz_vza ** 2))
#     cdef float HH = sigmahh * ShdwS * k ** 2 / 2 * np.exp(
#         -sigma ** 2 * (kz_iza ** 2 + kz_vza ** 2))
#
#     return VV, HH

# def shadowing_function(iza, vza, raa, rss):
#     if np.array_equal(vza, iza) and (np.allclose(np.all(raa), PI)):
#         ct = np.cos(iza) / np.sin(iza)
#         cts = np.cos(vza) / np.sin(vza)
#         rslp = rss
#         ctorslp = ct / np.sqrt(2) / rslp
#         ctsorslp = cts / np.sqrt(2) / rslp
#         shadf = 0.5 * (np.exp(-ctorslp ** 2) / np.sqrt(PI) / ctorslp - erf(ctorslp))
#         shadfs = 0.5 * (np.exp(-ctsorslp ** 2) / np.sqrt(PI) / ctsorslp - erf(ctsorslp))
#         ShdwS = 1 / (1 + shadf + shadfs)
#     else:
#         ShdwS = 1
#
#     return ShdwS


########################################################################################################################

#
# def emsv_integralfunc(float x, float y, float iza, double complex eps, double complex rv, double complex rh, float k,
#                       float kl, float ks,
#                       double complex sq, corrfunc, float corrlength, int pol):
#     cdef int nr
#     cdef np.ndarray wn, expwn, gauswn, svv, shh
#     cdef float wnn, vv, hv, ref, hh, vh
#     cdef double complex Fvv, Fhv, Fvvs, Fhvs, Ivv, Ihv, Ihh, Ivh
#
#     cdef float error = 1.0e3
#     cdef double complex sqs = np.sqrt(eps - sin(x) ** 2)
#     cdef double complex rc = (rv - rh) / 2
#     cdef double complex tv = 1 + rv
#     cdef double complex th = 1 + rh
#
#     # -- calc coefficients for surface correlation spectra
#     cdef float wvnb = k * np.sqrt(sin(iza) ** 2 - 2 * sin(iza) * sin(x) * cos(y) + sin(x) ** 2)
#
#     try:
#         nr = len(x)
#     except (IndexError, TypeError):
#         nr = 1
#
#     # -- calculate number of spectral components needed
#     cdef int n_spec = 1
#     while error > 1.0e-3:
#         n_spec = n_spec + 1
#         #   error = (ks2 *(cs + css)**2 )**n_spec / factorial(n_spec)
#         # ---- in this case we will use the smallest ths to determine the number of
#         # spectral components to use.  It might be more than needed for other angles
#         # but this is fine.  This option is used to simplify calculations.
#         error = (ks ** 2 * (cos(iza) + cos(x)) ** 2) ** n_spec / factorial(n_spec)
#         error = np.min(error)
#     # -- calculate expressions for the surface spectra
#
#     if corrfunc == 1:
#         wn = np.zeros([n_spec, nr])
#
#         for n in range(n_spec):
#             wn[n, :] = (n + 1) * kl ** 2 / ((n + 1) ** 2 + (wvnb * corrlength) ** 2) ** 1.5
#
#     if corrfunc == 2:
#         wn = np.zeros([n_spec, nr])
#
#         for n in range(n_spec):
#             wn[n, :] = 0.5 * kl ** 2 / (n + 1) * np.exp(-(wvnb * corrlength) ** 2 / (4 * (n + 1)))
#
#     if corrfunc == 3:
#         expwn = np.zeros([n_spec, nr])
#         gauswn = np.zeros([n_spec, nr])
#
#         for n in range(n_spec):
#             expwn[n, :] = (n + 1) * kl ** 2 / ((n + 1) ** 2 + (wvnb * corrlength) ** 2) ** 1.5
#             gauswn[n, :] = 0.5 * kl ** 2 / (n + 1) * np.exp(-(wvnb * corrlength) ** 2 / (4 * (n + 1)))
#
#         wn = expwn / gauswn
#
#     # -- calculate fpq!
#
#     cdef float ff = 2 * (sin(iza) * sin(x) - (1 + cos(iza) * cos(x)) * cos(y)) / (
#             cos(iza) + cos(x))
#
#     cdef double complex fvv = rv * ff
#     cdef double complex fhh = -rh * ff
#
#     cdef double complex fvh = -2 * rc * sin(y)
#     # cdef double complex fhv = 2 * rc * sin(y)
#
#     # -- calculate Fpq and Fpqs -----
#     cdef double complex fhv = sin(iza) * (sin(x) - sin(iza) * cos(y)) / (cos(iza) ** 2 * cos(x))
#     cdef double complex T = (sq * (cos(iza) + sq) + cos(iza) * (eps * cos(iza) + sq)) / (
#             eps * cos(iza) * (cos(iza) + sq) + sq * (eps * cos(iza) + sq))
#
#     cdef double complex cm2 = cos(x) * sq / cos(iza) / sqs - 1
#     cdef float ex = np.exp(-ks ** 2 * cos(iza) * cos(x))
#     cdef float de = 0.5 * np.exp(-ks ** 2 * (cos(iza) ** 2 + cos(x) ** 2))
#
#     if pol == 1:
#         Fvv = (eps - 1) * sin(iza) ** 2 * tv ** 2 * fhv / eps ** 2
#         Fhv = (T * sin(iza) * sin(iza) - 1. + cos(iza) / cos(x) + (
#                 eps * T * cos(iza) * cos(x) * (
#                 eps * T - sin(iza) * sin(iza)) - sq * sq) / (
#                        T * eps * sq * cos(x))) * (1 - rc * rc) * sin(y)
#
#         Fvvs = -cm2 * sq * tv ** 2 * (
#                 cos(y) - sin(iza) * sin(x)) / cos(
#             iza) ** 2 / eps - cm2 * sqs * tv ** 2 * cos(y) / eps - (
#                        cos(x) * sq / cos(iza) / sqs / eps - 1) * sin(
#             x) * tv ** 2 * (
#                        sin(iza) - sin(x) * cos(y)) / cos(iza)
#         Fhvs = -(sin(x) * sin(x) / T - 1 + cos(x) / cos(iza) + (
#                 cos(iza) * cos(x) * (
#                 1 - sin(x) * sin(x) * T) - T * T * sqs * sqs) / (
#                          T * sqs * cos(iza))) * (1 - rc * rc) * sin(y)
#
#         # -- calculate the bistatic field coefficients ---
#
#         svv = np.zeros([n_spec, nr])
#         for n in range(n_spec):
#             Ivv = fvv * ex * (ks * (cos(iza) + cos(x))) ** (n + 1) + (
#                     Fvv * (ks * cos(x)) ** (n + 1) + Fvvs * (ks * cos(iza)) ** (n + 1)) / 2
#             Ihv = fhv * ex * (ks * (cos(iza) + cos(x))) ** (n + 1) + (
#                     Fhv * (ks * cos(x)) ** (n + 1) + Fhvs * (ks * cos(iza)) ** (n + 1)) / 2
#
#         wnn = wn[n, :] / factorial(n + 1)
#         vv = wnn * (abs(Ivv)) ** 2
#         hv = wnn * (abs(Ihv)) ** 2
#         svv[n, :] = (de * (vv + hv) * sin(x) * (1 / cos(iza))) / (4 * PI)
#
#         ref = np.sum([svv])  # adding all n terms stores in different rows
#
#     if pol == 2:
#         Fhh = -(eps - 1) * th ** 2 * fhv
#         Fvh = (sin(iza) * sin(iza) / T - 1. + cos(iza) / cos(x) + (
#                 cos(iza) * cos(x) * (
#                 1 - sin(iza) * sin(iza) * T) - T * T * sq * sq) / (
#                        T * sq * cos(x))) * (1 - rc * rc) * sin(y)
#
#         Fhhs = cm2 * sq * th ** 2 * (
#                 cos(y) - sin(iza) * sin(x)) / cos(
#             iza) ** 2 + cm2 * sqs * th ** 2 * cos(y) + cm2 * sin(x) * th ** 2 * (
#                        sin(iza) - sin(x) * cos(y)) / cos(iza)
#         Fvhs = -(T * sin(x) * sin(x) - 1 + cos(x) / cos(iza) + (
#                 eps * T * cos(iza) * cos(x) * (
#                 eps * T - sin(x) * sin(x)) - sqs * sqs) / (
#                          T * eps * sqs * cos(iza))) * (1 - rc * rc) * sin(y)
#
#         shh = np.zeros([n_spec, nr])
#         for n in range(n_spec):
#             Ihh = fhh * ex * (ks * (cos(iza) + cos(x))) ** (n + 1) + (
#                     Fhh * (ks * cos(x)) ** (n + 1) + Fhhs * (ks * cos(iza)) ** (n + 1)) / 2
#             Ivh = fvh * ex * (ks * (cos(iza) + cos(x))) ** (n + 1) + (
#                     Fvh * (ks * cos(x)) ** (n + 1) + Fvhs * (ks * cos(iza)) ** (n + 1)) / 2
#
#         wnn = wn[n, :] / factorial(n + 1)
#         hh = wnn * (abs(Ihh)) ** 2
#         vh = wnn * (abs(Ivh)) ** 2
#         (2 * (3 + 4) * sin(5) * 1 / cos(6)) / (PI * 4)
#         shh[n, :] = (de * (hh + vh) * sin(x) * (1 / cos(iza))) / (4 * PI)
#
#         ref = np.sum([shh])
#
#     return ref
#
# cdef calc_iem_ems(float iza, float k, float sigma, float corrlength, double complex eps, int corrfunc_ems):
#     cdef float ks = k * sigma
#     cdef float kl = k * corrlength
#
#     # -- calculation of reflection coefficients
#     cdef double complex sq = np.sqrt(eps - np.sin(iza) ** 2)
#
#     cdef double complex rv = (eps * np.cos(iza) - sq) / (
#             eps * np.cos(iza) + sq)
#
#     cdef double complex rh = (np.cos(iza) - sq) / (np.cos(iza) + sq)
#
#     cdef float refv = dblquad(emsv_integralfunc, 0, PI / 2, lambda x: 0, lambda x: PI,
#                               args=(iza, eps, rv, rh, k, kl, ks, sq, corrfunc_ems, corrlength, 1))[0]
#
#     cdef float refh = dblquad(emsv_integralfunc, 0, PI / 2, lambda x: 0, lambda x: PI,
#                               args=(
#                                   iza, eps, rv, rh, k, kl, ks, sq, corrfunc_ems, corrlength,
#                                   2))[0]
#
#     cdef float VV = 1 - refv - np.exp(-ks ** 2 * np.cos(iza) * np.cos(iza)) * (
#         abs(rv)) ** 2
#     cdef float HH = 1 - refh - np.exp(-ks ** 2 * np.cos(iza) * np.cos(iza)) * (
#         abs(rh)) ** 2
#
#     return VV, HH
