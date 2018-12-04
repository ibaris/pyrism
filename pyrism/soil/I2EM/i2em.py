# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from pyrism.core.iemauxil import i2em_wrapper, wvnb_wrapper, TS_wrapper  # , calc_iem_ems_wrapper
from respy import Angles, dB, BRDF, BRF, align_all, asarrays, convert_frequency
from respy import compute_wavenumber as wavenumber
from respy import EMW
from .core import (exponential, gaussian, xpower, mixed)
from ...auxil import SoilResult, EmissivityResult
import warnings


class I2EM(Angles):

    def __init__(self, iza, vza, raa, frequency, eps, corrlength, sigma, n=10, corrfunc='exponential', emissivity=False,
                 angle_unit='DEG', frequency_unit='GHz'):

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
        # Align Input Parameter ----------------------------------------------------------------------------------------
        iza, vza, raa, frequency, corrlength, sigma = align_all((iza, vza, raa, frequency, corrlength, sigma),
                                                                dtype=np.double)
        _, eps = align_all((iza, eps), dtype=np.complex)
        _, n = align_all((iza, n), dtype=np.intc)

        if angle_unit == 'DEG':  # This is the same procedure as in the Ulaby codes.
            iza += 0.5729577951308232
        else:
            iza += 0.01

        super(I2EM, self).__init__(iza=iza, vza=vza, raa=raa, normalize=False, nbar=0.0,
                                   angle_unit=angle_unit)

        # Assign Frequency and Roughness -------------------------------------------------------------------------------
        sl = sigma / corrlength  # Roughness Parameter

        self.EMW = EMW(input=frequency, unit=frequency_unit, output='cm')
        frequency = self.EMW.frequency
        k = self.EMW.k0

        # < Check Validity of I2EM > ------------
        if any(sl.real > 0.25):
            warnings.warn("I2EM is valid for sigma/corrlength < 0.25. The actual ratio is: {0}".format(str(sl)))
        if any(frequency.real > 4.5):
            warnings.warn("I2EM is valid for frequency < 4.5. The actual frequency is: {0}".format(str(frequency.real)))

        # Define Angle Parameter ---------------------------------------------------------------------------------------
        kz_iza = k * np.cos(self.iza)
        kz_vza = k * np.cos(self.vza)
        phi = np.zeros_like(iza)

        # Calculations -------------------------------------------------------------------------------------------------
        # < Correlation Function > ------------
        corrfunc = self.__set_corrfunc(corrfunc)

        # < I2EM > ------------
        VV, HH = i2em_wrapper(k=k, kz_iza=kz_iza, kz_vza=kz_vza, iza=self.iza, vza=self.vza, raa=self.raa, phi=phi,
                              eps=eps, corrlength=corrlength, sigma=sigma, corrfunc=corrfunc, n=n)

        # # calculate I2EM BSC
        # VV_list, HH_list = list(), list()
        # for i in range(len(self.iza)):
        #     VV, HH = calc_i2em_auxil(k[i], kz_iza[i], kz_vza[i], izac[i], vzac[i], raac[i], phi, eps[i], corrlengthc[i],
        #                              sigmac[i], corrfunc_, n)
        #
        #     VV_list.append(VV)
        #     HH_list.append(HH)
        #
        # VV, HH = asarrays((VV_list, HH_list))

        # Store data
        self.I, self.BRF, self.BSC = self.__store(VV.base, HH.base)

        if emissivity:
            pass
            # ems = I2EM.Emissivity(iza, frequency, eps, corrlength, sigma, corrfunc=corrfunc, angle_unit=angle_unit)
            # self.EMS = ems.EMS

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

        BSC = SoilResult(array=np.array([[VV], [HH]]),
                         arraydB=np.array([[dB(VV)], [dB(HH)]]),
                         VV=VV,
                         HH=HH,
                         VVdB=dB(VV),
                         HHdB=dB(HH))

        I = SoilResult(array=np.array([[BRDF(VV, self.vza)], [BRDF(HH, self.vza)]]),
                       arraydB=np.array([[dB(BRDF(VV, self.vza))], [dB(BRDF(HH, self.vza))]]),
                       VV=BRDF(VV, self.vza),
                       HH=BRDF(HH, self.vza))

        BRF_ = SoilResult(array=np.array([[BRF(I.VV)], [BRF(I.HH)]]),
                          arraydB=np.array([[dB(BRF(I.VV))], [dB(BRF(I.HH))]]),
                          VV=BRF(I.VV),
                          HH=BRF(I.HH))

        return I, BRF_, BSC

    # class Emissivity(Angles):
    #
    #     def __init__(self, iza, frequency, eps, corrlength, sigma, corrfunc='exponential', angle_unit='DEG',
    #                  frequency_unit='GHz'):
    #         """
    #         This Class calculates the emission from rough surfaces using the
    #         I2EM Model.
    #
    #         Parameters
    #         ----------
    #         iza : int, float or ndarray
    #             Incidence (iza) and scattering (vza) zenith angle, as well as relative azimuth (raa) angle.
    #         angle_unit : {'DEG', 'RAD'}, optional
    #             * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
    #             * 'RAD': All input angles (iza, vza, raa) are in [RAD].
    #         frequency : int, float or ndarray
    #             RADAR Frequency (GHz).
    #         eps : int, float or ndarray
    #             Complex dielectric constant of soil.
    #         corrlength : int, float or ndarray
    #             Correlation length (cm).
    #         sigma : int, float or ndarray
    #             RMS Height (cm)
    #         corrfunc : {'exponential', 'gaussian', 'mixed'}, optional
    #             Correlation distribution functions. The `mixed` correlation function is the result of the division of
    #             gaussian correlation function with exponential correlation function. Default is 'exponential'.
    #
    #         Returns
    #         -------
    #         For attributes see also core.Kernel and core.EmissivityResult.
    #
    #         See Also
    #         --------
    #         pyrism.core.EmissivityResult
    #         """
    #
    #         vza = np.zeros_like(iza)
    #         raa = vza
    #
    #         eps = np.asarray(eps).flatten()
    #         iza, vza, raa, frequency, corrlength, sigma = asarrays((iza, vza, raa, frequency, corrlength, sigma))
    #
    #         iza, vza, raa, frequency, corrlength, sigma, eps = align_all(
    #             (iza, vza, raa, frequency, corrlength, sigma, eps))
    #
    #         super(I2EM.Emissivity, self).__init__(iza, vza, raa, normalize=False, nbar=0.0, angle_unit=angle_unit)
    #
    #         frequency /= 1e9
    #
    #         k = wavenumber(frequency, unit=frequency_unit, output='cm')
    #         corrfunc = self.__set_corrfunc(corrfunc)
    #
    #         # calculate I2EM BSC
    #         V_list, H_list = list(), list()
    #         for i in range(len(self.iza)):
    #             V, H = calc_iem_ems_wrapper(self.iza[i], k[i], sigma[i], corrlength[i], eps[i], corrfunc)
    #
    #             V_list.append(V)
    #             H_list.append(H)
    #
    #         V, H = asarrays((V_list, H_list))
    #
    #         self.EMS = self.__store(V, H)
    #
    #     def __set_corrfunc(self, corrfunc):
    #         if corrfunc is 'exponential':
    #             corrfunc = 1
    #         elif corrfunc is 'gaussian':
    #             corrfunc = 2
    #         elif corrfunc is 'mixed':
    #             corrfunc = 3
    #         else:
    #             raise ValueError("The parameter corrfunc must be 'exponential', 'gaussian' or 'mixed'")
    #
    #         return corrfunc
    #
    #     def __store(self, V, H):
    #
    #         EMS = EmissivityResult(array=np.array([[V], [H]]),
    #                                arraydB=np.array([[dB(V)], [dB(H)]]),
    #                                V=V,
    #                                H=H,
    #                                VdB=dB(V),
    #                                HdB=dB(H))
    #
    #         return EMS
