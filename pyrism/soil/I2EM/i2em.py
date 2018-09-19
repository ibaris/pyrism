# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from pyrism.core.iemauxil import calc_i2em_auxil, calc_iem_ems_wrapper
from radarpy import Angles, dB, BRDF, BRF, align_all, asarrays

from .core import (exponential, gaussian, xpower, mixed)
from ...auxil import SoilResult, EmissivityResult

class I2EM(Angles):

    def __init__(self, iza, vza, raa, frequency, eps, corrlength, sigma, n=10, corrfunc='exponential', emissivity=False,
                 angle_unit='DEG'):

        """
        RADAR Surface Scatter Based Kernel (I2EM). Compute BSC VV and
        BSC HH and the emissivity for single-scale random surface for
        Bi and Mono-static acquisitions (:cite:`Ulaby.2015` and :cite:`Ulaby.2015b`).

        Parameters
        ----------
        iza, vza, raa : int, float or ndarray
            Incidence (iza) and scattering (vza) zenith angle, as well as relative azimuth (raa) angle.
        angle_unit : {'DEG', 'RAD'}, optional
            * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
            * 'RAD': All input angles (iza, vza, raa) are in [RAD].
        frequency : int, float or ndarray
            RADAR Frequency (GHz).
        diel_constant : int, float or ndarray
            Complex dielectric constant of soil.
        corrlength : int, float or ndarray
            Correlation length (cm).
        sigma : int, float or ndarray
            RMS Height (cm)
        n : int (default = 10), optinal
            Coefficient needed for x-power and x-exponential
            correlation function.
        corrfunc : {'exponential', 'gaussian', 'xpower', 'mixed'}, optional
            Correlation distribution functions. The `mixed` correlation function is the result of the division of
            gaussian correlation function with exponential correlation function. Default is 'exponential'.
        emissivity : bool
            If True I2EM calculates automatically the emission of the surface. Note, if corrfunc is 'xpower' the
            'exponential' correlation function is used.

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

        eps = np.asarray(eps).flatten()
        iza, vza, raa, frequency, corrlength, sigma = asarrays((iza, vza, raa, frequency, corrlength, sigma))

        iza, vza, raa, frequency, corrlength, sigma, eps = align_all((iza, vza, raa, frequency, corrlength, sigma, eps))

        super(I2EM, self).__init__(iza=iza.real, vza=vza.real, raa=raa.real, normalize=False, nbar=0.0,
                                   angle_unit=angle_unit)

        # Setup variables
        k = 2 * np.pi * frequency / 30
        kz_iza = k * np.cos(self.iza + 0.01)
        kz_vza = k * np.cos(self.vza)
        phi = 0. + 0.j
        corrfunc_ = self.__set_corrfunc(corrfunc)

        # Setup temporal complex variables. This is due to a incompatibility with complex numbers and cython.
        izac = self.iza.astype(np.complex)
        vzac = self.vza.astype(np.complex)
        raac = self.raa.astype(np.complex)
        sigmac = sigma + 0.j
        corrlengthc = corrlength + 0.j

        # calculate I2EM BSC
        VV_list, HH_list = list(), list()
        for i in range(len(self.iza)):
            VV, HH = calc_i2em_auxil(k[i], kz_iza[i], kz_vza[i], izac[i], vzac[i], raac[i], phi, eps[i], corrlengthc[i],
                                     sigmac[i], corrfunc_, n)

            VV_list.append(VV)
            HH_list.append(HH)

        VV, HH = asarrays((VV_list, HH_list))

        # Store data
        self.I, self.BRF, self.BSC = self.__store(VV, HH)

        if emissivity:
            ems = I2EM.Emissivity(iza, frequency, eps, corrlength, sigma, corrfunc=corrfunc, angle_unit=angle_unit)
            self.EMS = ems.EMS

    def __set_corrfunc(self, corrfunc):
        if corrfunc is 'exponential':
            corrfunc = exponential

        elif corrfunc is 'gaussian':
            corrfunc = gaussian

        elif corrfunc is 'xpower':
            corrfunc = xpower

        elif corrfunc is 'mixed':
            corrfunc = mixed

        else:
            raise ValueError("The parameter corrfunc must be 'exponential', 'gaussian', 'xpower' or 'mixed'")

        return corrfunc

    def __store(self, VV, HH):

        BSC = SoilResult(VV=VV,
                         HH=HH,
                         VVdB=dB(VV),
                         HHdB=dB(HH),
                         ISO=(VV + HH) / 2,
                         ISOdB=dB((VV + HH) / 2))

        I = SoilResult(VV=BRDF(VV, self.vza),
                       HH=BRDF(HH, self.vza),
                       ISO=BRDF(BSC.ISO, self.vza))

        BRF_ = SoilResult(VV=BRF(I.VV),
                          HH=BRF(I.HH),
                          ISO=BRF(I.ISO))

        return I, BRF_, BSC

    class Emissivity(Angles):

        def __init__(self, iza, frequency, eps, corrlength, sigma, corrfunc='exponential', angle_unit='DEG'):
            """
            This Class calculates the emission from rough surfaces using the
            I2EM Model.

            Parameters
            ----------
            iza : int, float or ndarray
                Incidence (iza) and scattering (vza) zenith angle, as well as relative azimuth (raa) angle.
            angle_unit : {'DEG', 'RAD'}, optional
                * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
                * 'RAD': All input angles (iza, vza, raa) are in [RAD].
            frequency : int, float or ndarray
                RADAR Frequency (GHz).
            eps : int, float or ndarray
                Complex dielectric constant of soil.
            corrlength : int, float or ndarray
                Correlation length (cm).
            sigma : int, float or ndarray
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

            vza = np.zeros_like(iza)
            raa = vza

            eps = np.asarray(eps).flatten()
            iza, vza, raa, frequency, corrlength, sigma = asarrays((iza, vza, raa, frequency, corrlength, sigma))

            iza, vza, raa, frequency, corrlength, sigma, eps = align_all(
                (iza, vza, raa, frequency, corrlength, sigma, eps))

            super(I2EM.Emissivity, self).__init__(iza, vza, raa, normalize=False, nbar=0.0, angle_unit=angle_unit)

            fr = frequency / 1e9
            k = 2 * np.pi * fr / 30
            corrfunc = self.__set_corrfunc(corrfunc)

            # calculate I2EM BSC
            V_list, H_list = list(), list()
            for i in range(len(self.iza)):
                V, H = calc_iem_ems_wrapper(self.iza[i], k[i], sigma[i], corrlength[i], eps[i], corrfunc)

                V_list.append(V)
                H_list.append(H)

            V, H = asarrays((V_list, H_list))

            self.EMS = self.__store(V, H)

        def __set_corrfunc(self, corrfunc):
            if corrfunc is 'exponential':
                corrfunc = 1
            elif corrfunc is 'gaussian':
                corrfunc = 2
            elif corrfunc is 'mixed':
                corrfunc = 3
            else:
                raise ValueError("The parameter corrfunc must be 'exponential', 'gaussian' or 'mixed'")

            return corrfunc

        def __store(self, V, H):

            EMS = EmissivityResult(array=np.array([[V], [H]]),
                                   arraydB=np.array([[dB(V)], [dB(H)]]),
                                   V=V,
                                   H=H,
                                   VdB=dB(V),
                                   HdB=dB(H))

            EMS['ISO'] = (EMS.V + EMS.H) / 2
            EMS['ISOdB'] = dB(EMS.ISO)

            return EMS
