# -*- coding: utf-8 -*-
from __future__ import division

import sys

import numpy as np
from scipy.integrate import dblquad
from scipy.misc import factorial

from .core import (exponential, gaussian, xpower, mixed, reflection_coefficients, exponential_ems, gaussian_ems,
                   mixed_ems, r_transition, Ra_integration,
                   biStatic_coefficient, Ipp, shadowing_function, emsv_integralfunc)

from radarpy import Angles, dB, BRDF, BRF, align_all, asarrays, stacks
from ...auxil import SoilResult

# python 3.6 comparability
if sys.version_info < (3, 0):
    srange = xrange
else:
    srange = range


class I2EM(Angles):

    def __init__(self, iza, vza, raa, nbar=0.0, angle_unit='DEG', frequency=None, eps=None,
                 corrlength=None, sigma=None, n=10, corrfunc='exponential'):

        """
        RADAR Surface Scatter Based Kernel (I2EM). Compute BSC VV and
        BSC HH and the emissivity for single-scale random surface for
        Bi and Mono-static acquisitions (:cite:`Ulaby.2015` and :cite:`Ulaby.2015b`).

        Parameters
        ----------
        iza, vza, raa : int, float or ndarray
            Incidence (iza) and scattering (vza) zenith angle, as well as relative azimuth (raa) angle.
        normalize : boolean, optional
            Set to 'True' to make kernels 0 at nadir view illumination. Since all implemented kernels are normalized
            the default value is False.
        nbar : float, optional
            The sun or incidence zenith angle at which the isotropic term is set
            to if normalize is True. The default value is 0.0.
        angle_unit : {'DEG', 'RAD'}, optional
            * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
            * 'RAD': All input angles (iza, vza, raa) are in [RAD].
        frequency : int or float
            RADAR Frequency (GHz).
        diel_constant : int or float
            Complex dielectric constant of soil.
        corrlength : int or float
            Correlation length (cm).
        sigma : int or float
            RMS Height (cm)
        n : int (default = 10), optinal
            Coefficient needed for x-power and x-exponential
            correlation function.
        corrfunc : {'exponential', 'gaussian', 'xpower', 'mixed'}, optional
            Correlation distribution functions. The `mixed` correlation function is the result of the division of
            gaussian correlation function with exponential correlation function. Default is 'exponential'.

        Returns
        -------
        For more attributes see also pyrism.core.Kernel and pyrism.core.ReflectanceResult.

        See Also
        --------
        I2EM.Emissivity
        pyrism.core.Kernel
        pyrism.core.ReflectanceResult


        Note
        ----
        The model is constrained to realistic surfaces with
        (rms height / correlation length) ≤ 0.25.
        Hot spot direction is vza == iza and raa = 0.0

        """

        super(I2EM, self).__init__(iza, vza, raa, normalize=False, nbar=nbar, angle_unit=angle_unit)

        if corrfunc is 'exponential':
            self.corrfunc = exponential
        elif corrfunc is 'gaussian':
            self.corrfunc = gaussian
        elif corrfunc is 'xpower':
            self.corrfunc = xpower
        elif corrfunc is 'mixed':
            self.corrfunc = mixed
        else:
            raise ValueError("The parameter corrfunc must be 'exponential', 'gaussian', 'xpower' or 'mixed'")

        # frequency, eps, corrlength, sigma = asarrays((frequency, eps, corrlength, sigma))
        # iza, frequency, eps, corrlength, sigma = align_all((self.iza, frequency, eps, corrlength, sigma))

        # self.iza = iza.real

        self.eps = eps
        self.corrlength = corrlength
        self.sigma = sigma
        self.k = 2 * np.pi * frequency / 30
        kz_iza = self.k * np.cos(self.iza + 0.01)
        kz_vza = self.k * np.cos(self.vza)
        self.phi = 0

        rt, Rvi, Rhi, wvnb, Ts = reflection_coefficients(self.k, self.iza, self.vza, self.raa, self.phi,
                                                         eps, sigma)

        Wn, rss = self.corrfunc(sigma, corrlength, wvnb, Ts, n=n)

        Tf, Rv0, Rh0 = r_transition(self.k, self.iza, self.vza, sigma, eps, Wn, Ts)

        RaV, RaH = Ra_integration(self.iza, sigma, corrlength, eps)

        fvv, fhh, Rvt, Rht = biStatic_coefficient(self.iza, self.vza, self.raa, Rvi, Rv0, Rhi, Rh0, RaV, RaH, Tf)

        Ivv, Ihh = Ipp(self.iza, self.vza, self.raa, self.phi, Rvi, Rhi, eps, self.k, kz_iza, kz_vza, fvv, fhh, sigma,
                       Ts)

        ShdwS = shadowing_function(self.iza, self.vza, self.raa, rss)

        self.VV, self.HH, self.VVdB, self.HHdB = self.__sigma_nought(Ts, Wn, Ivv, Ihh, ShdwS, self.k, kz_iza, kz_vza,
                                                                     self.sigma)

        self.EVV, self.EHH, self.EVVdB, self.EHHdB = self.__sigma_nought(Ts, Wn, Ivv, Ihh, ShdwS, self.k, kz_iza,
                                                                         kz_vza,
                                                                         self.sigma)

        self.__store()

    def __sigma_nought(self, Ts, Wn, Ivv, Ihh, ShdwS, k, kz_iza, kz_vza, sigma):

        sigmavv = 0
        sigmahh = 0
        for i in srange(Ts):
            i += 1
            a0 = Wn[i - 1] / factorial(i) * sigma ** (2 * i)

            sigmavv = sigmavv + np.abs(Ivv[i - 1]) ** 2 * a0
            sigmahh = sigmahh + np.abs(Ihh[i - 1]) ** 2 * a0

        VV = sigmavv * ShdwS * k ** 2 / 2 * np.exp(
            -sigma ** 2 * (kz_iza ** 2 + kz_vza ** 2))
        HH = sigmahh * ShdwS * k ** 2 / 2 * np.exp(
            -sigma ** 2 * (kz_iza ** 2 + kz_vza ** 2))

        with np.errstate(invalid='ignore'):
            VVdB = dB(np.asarray(VV, dtype=np.float))
            HHdB = dB(np.asarray(HH, dtype=np.float))

        return VV, HH, VVdB, HHdB

    def __emissivity(self):

        # fr = self.freq / 1e9
        # k =
        ks = self.k * self.sigma  # roughness parameter
        kl = self.k * self.corrlength

        # -- calculation of reflection coefficients
        sq = np.sqrt(self.eps - np.sin(self.iza) ** 2)

        rv = (self.eps * np.cos(self.iza) - sq) / (
                self.eps * np.cos(self.iza) + sq)

        rh = (np.cos(self.iza) - sq) / (np.cos(self.iza) + sq)

        refv = dblquad(emsv_integralfunc, 0, np.pi / 2, lambda x: 0, lambda x: np.pi,
                       args=(
                           self.iza, self.eps, self.rv, self.rh, self.k, kl, ks, sq, self.corrfunc, self.corrlength,
                           'vv'))

        refh = dblquad(emsv_integralfunc, 0, np.pi / 2, lambda x: 0, lambda x: np.pi,
                       args=(
                           self.iza, self.eps, self.rv, self.rh, self.k, kl, ks, sq, self.corrfunc, self.corrlength,
                           'hh'))

        VV = 1 - refv[0] - np.exp(-ks ** 2 * np.cos(self.iza) * np.cos(self.iza)) * (
            abs(rv)) ** 2
        HH = 1 - refh[0] - np.exp(-ks ** 2 * np.cos(self.iza) * np.cos(self.iza)) * (
            abs(rh)) ** 2

        with np.errstate(invalid='ignore'):
            VVdB = dB(np.asarray(VV, dtype=np.float))
            HHdB = dB(np.asarray(HH, dtype=np.float))

        return VV, HH, VVdB, HHdB

    def __store(self):

        self.BSC = SoilResult(VV=self.VV,
                              HH=self.HH,
                              VVdB=self.VVdB,
                              HHdB=self.HHdB,
                              VH=np.zeros_like(self.VV),
                              HV=np.zeros_like(self.VV),
                              VHdB=np.zeros_like(self.VV),
                              HVdB=np.zeros_like(self.VV))

        self.BSC['ISO'] = (self.BSC.VV + self.BSC.HH) / 2
        self.BSC['ISOdB'] = dB(self.BSC.ISO)

        self.BSC['array'] = stacks((self.BSC.ISO, self.BSC.VV, self.BSC.HH, self.BSC.VH, self.BSC.HV))
        self.BSC['arraydB'] = stacks((self.BSC.ISOdB, self.BSC.VVdB, self.BSC.HHdB, self.BSC.VHdB, self.BSC.HVdB))

        self.I = SoilResult(VV=BRDF(self.VV, self.vza),
                            HH=BRDF(self.HH, self.vza),
                            VH=np.zeros_like(self.VV),
                            HV=np.zeros_like(self.HH))

        self.I['ISO'] = (self.I.VV + self.I.HH) / 2

        self.I['array'] = stacks((self.I.ISO, self.I.VV, self.I.HH, self.I.VH, self.I.HV))

        self.BRF = SoilResult(VV=BRF(self.I.VV),
                              HH=BRF(self.I.HH),
                              VH=np.zeros_like(self.VV),
                              HV=np.zeros_like(self.HH))

        self.BRF['ISO'] = (self.BRF.VV + self.BRF.HH) / 2

        self.BRF['array'] = stacks((self.BRF.ISO, self.BRF.VV, self.BRF.HH, self.BRF.VH, self.BRF.HV))

        self.E = SoilResult(VV=self.EVV,
                            HH=self.EHH,
                            VH=np.zeros_like(self.EVV),
                            HV=np.zeros_like(self.VV))

        self.E['ISO'] = (self.E.VV + self.E.HH) / 2

        self.E['array'] = stacks((self.E.ISO, self.E.VV, self.E.HH, self.E.VH, self.E.HV))

    class Emissivity(Angles):

        def __init__(self, iza, vza, raa, normalize=False, nbar=0.0, angle_unit='DEG',
                     frequency=1.26, eps=10 + 1j, corrlength=10, sigma=0.3, corrfunc='exponential'):
            """
            This Class calculates the emission from rough surfaces using the
            I2EM Model.

            Parameters
            ----------
             iza, vza, raa : int, float or ndarray
                 Incidence (iza) and scattering (vza) zenith angle, as well as relative azimuth (raa) angle.
             normalize : boolean, optional
                 Set to 'True' to make kernels 0 at nadir view illumination. Since all implemented kernels are normalized
                 the default value is False.
             nbar : float, optional
                 The sun or incidence zenith angle at which the isotropic term is set
                 to if normalize is True. The default value is 0.0.
             angle_unit : {'DEG', 'RAD'}, optional
                 * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
                 * 'RAD': All input angles (iza, vza, raa) are in [RAD].
             frequency : int or float
                 RADAR Frequency (GHz).
             eps : int or float
                 Complex dielectric constant of soil.
             corrlength : int or float
                 Correlation length (cm).
             sigma : int or float
                 RMS Height (cm)
             corrfunc : {'exponential', 'gaussian', 'mixed'}, optional
                 Correlation distribution functions. The `mixed` correlation function is the result of the division of
                 gaussian correlation function with exponential correlation function. Default is 'exponential'.

            Returns
            -------
            For attributes see also core.Kernel and core.EmissivityResult.

            See Also
            --------
            pyrism.core.EmissivityResult
            """

            super(I2EM.Emissivity, self).__init__(iza, vza, raa, normalize, nbar, angle_unit)

            if corrfunc is 'exponential':
                self.corrfunc = exponential_ems
            elif corrfunc is 'gaussian':
                self.corrfunc = gaussian_ems
            elif corrfunc is 'mixed':
                self.corrfunc = mixed_ems
            else:
                raise ValueError("The parameter corrfunc must be 'exponential', 'gaussian' or 'mixed'")

            self.eps = eps
            self.corrlength = corrlength  # in cm
            self.sigma = sigma  # in cm
            self.freq = frequency

            fr = self.freq / 1e9

            self.k = 2 * np.pi * fr / 30  # wavenumber in free space.  Speed of light is in cm/sec

            ks = self.k * self.sigma  # roughness parameter
            kl = self.k * self.corrlength

            # -- calculation of reflection coefficients
            sq = np.sqrt(self.eps - np.sin(self.iza) ** 2)

            rv = (eps * np.cos(self.iza) - sq) / (
                    eps * np.cos(self.iza) + sq)
            rh = (np.cos(self.iza) - sq) / (np.cos(self.iza) + sq)

            self.VV, self.HH, self.VVdB, self.HHdB = self.__calc(self.iza, eps, rv, rh, self.k, kl, ks, sq,
                                                                 self.corrfunc, corrlength)

            self.EMS = SoilResult(array=np.array([[self.VV], [self.HH]]),
                                  arraydB=np.array([[dB(self.VV)], [dB(self.HH)]]),
                                  VV=self.VV,
                                  HH=self.HH,
                                  VVdB=dB(self.VV),
                                  HHdB=dB(self.HH))

        def __calc(self, iza, eps, rv, rh, k, kl, ks, sq, corrfunc, corrlength):
            refv = dblquad(emsv_integralfunc, 0, np.pi / 2, lambda x: 0, lambda x: np.pi,
                           args=(iza, eps, rv, rh, k, kl, ks, sq, corrfunc, corrlength, 'vv'))

            refh = dblquad(emsv_integralfunc, 0, np.pi / 2, lambda x: 0, lambda x: np.pi,
                           args=(iza, eps, rv, rh, k, kl, ks, sq, corrfunc, corrlength, 'hh'))

            VV = 1 - refv[0] - np.exp(-ks ** 2 * np.cos(iza) * np.cos(iza)) * (
                abs(rv)) ** 2
            HH = 1 - refh[0] - np.exp(-ks ** 2 * np.cos(iza) * np.cos(iza)) * (
                abs(rh)) ** 2

            with np.errstate(invalid='ignore'):
                VVdB = dB(np.asarray(VV, dtype=np.float))
                HHdB = dB(np.asarray(HH, dtype=np.float))

            return VV, HH, VVdB, HHdB
