# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""
from __future__ import division
from pyrism.cython_iem.auxil cimport compute_ShdwS
from pyrism.cython_iem.rspectrum cimport compute_wvnb, compute_TS, compute_Wn_rss
from pyrism.cython_iem.fresnel cimport  compute_rt, compute_Rxi, compute_Rx0
from pyrism.cython_iem.transition cimport compute_Ft, compute_Tf
from pyrism.cython_iem.bicoef cimport Rax_integration, compute_Rxt, compute_fxx
from pyrism.cython_iem.fxxyxx cimport compute_Fxxyxx
from pyrism.cython_iem.ipp cimport compute_IPP
from pyrism.cython_iem.sigma cimport compute_sigma_nought

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

cdef tuple compute_ixx(double[:] k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
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


    return Ivv, Ihh
