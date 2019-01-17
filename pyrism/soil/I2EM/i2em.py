# -*- coding: utf-8 -*-
from __future__ import division

import warnings

import numpy as np
from pyrism.core.iemauxil import i2em_wrapper, calc_iem_ems_wrapper
from respy import Angles, dB, BRDF, BRF, align_all, asarrays, RAD_TO_DEG, DEG_TO_RAD
from respy import EMW

from pyrism.auxil import SoilResult, EmissivityResult
from pyrism.soil.I2EM.core import (exponential, gaussian, xpower, mixed)


class I2EM(Angles):

    def __init__(self, iza, vza, raa, frequency, eps, corrlength, sigma, n=10, corrfunc='exponential',
                 angle_unit='DEG', frequency_unit='GHz', roughness_unit='m'):

        """
        RADAR Surface Scatter Based Kernel (I2EM). Compute BSC VV and
        BSC HH and the emissivity for single-scale random surface for
        Bi and Mono-static acquisitions (:cite:`Ulaby.2015` and :cite:`Ulaby.2015b`).

        Parameters
        ----------
        iza, vza, raa : int, float or ndarray
            Incidence (iza) and scattering (vza) zenith angle, as well as relative azimuth (raa) angle in
            {'DEG' or 'RAD'}. See parameter angle_unit.
        frequency : int, float or ndarray
            RADAR Frequency {'Hz', 'MHz', 'GHz', 'THz'} (see parameter frequency_unit).
        eps : int, float or ndarray
            Complex dielectric constant of soil in {'dm', 'nm', 'cm', 'mm', 'm', 'km', 'um'}. See parameter
            roughness_unit.
        corrlength : int, float or ndarray
            Correlation length in {'dm', 'nm', 'cm', 'mm', 'm', 'km', 'um'}. See parameter
            roughness_unit.
        sigma : int, float or ndarray
            RMS Height in {'dm', 'nm', 'cm', 'mm', 'm', 'km', 'um'}. See parameter
            roughness_unit.
        n : int (default = 10), optinal
            Coefficient needed for x-power and x-exponential correlation function.
        corrfunc : {'exponential', 'gaussian', 'xpower', 'mixed'}, optional
            Correlation distribution functions. The `mixed` correlation function is the result of the division of
            gaussian correlation function with exponential correlation function. Default is 'exponential'.
        frequency_unit : {'Hz', 'PHz', 'kHz', 'daHz', 'MHz', 'THz', 'hHz', 'GHz'}
            Unit of entered frequency. Default is 'GHz'.
        roughness_unit : {'dm', 'nm', 'cm', 'mm', 'm', 'km', 'um'}
            Unit of the radius in meter (m), centimeter (cm) or nanometer (nm).
        angle_unit : {'DEG', 'RAD'}, optional
            * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
            * 'RAD': All input angles (iza, vza, raa) are in [RAD].

        Attributes
        ----------
        iza, vza, raa, iaa, vaa, alpha, beta: array_like
            Incidence (iza) and scattering (vza) zenith angle, relative azimuth (raa) angle, incidence and viewing
            azimuth angle (ira, vra) in [RAD].
        izaDeg, vzaDeg, raaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg: array_like
            SIncidence (iza) and scattering (vza) zenith angle, relative azimuth (raa) angle, incidence and viewing
            azimuth angle (ira, vra) in [DEG].
        phi : array_like
            Relative azimuth angle in a range between 0 and 2pi.
        B, BDeg : array_like
            The result of (1/cos(vza)+1/cos(iza)).
        mui, muv : array_like
            Cosine of iza and vza in [RAD].
        nbar : float
            The sun or incidence zenith angle at which the isotropic term is set
            to if normalize is True. You can change this attribute within the class.
        dtype : numpy.dtype
            Desired data type of all values. This attribute is changeable.
        frequency : array_like
            Frequency. Access with respy.EMW.
        wavelength : array_like
            Wavelength. Access with respy.EMW.
        k0 : array_like
            Free space wavenumber in unit of wavelength_unit.
        frequency_unit : str
            Frequency unit. Access with respy.EMW.
        wavelength_unit : str
            Wavelength unit. This is the same as radius unit. Access with respy.EMW.
        len : int
            Length of elements.
        sigma : array_like
        corrlength : array_like
        corrfunc : callable
        n : int

        For more attributes see also pyrism.core.SoilResult.

        See Also
        --------
        I2EM.Emissivity
        pyrism.core.SoilResult

        Note
        ----
        The model is constrained to realistic surfaces with (rms height / correlation length) ≤ 0.25.
        Hot spot direction is vza == iza and raa = 0.0

        """
        # Define angles and align data -----------------------------------------------------------------------------
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

        # Define Frequency -----------------------------------------------------------------------------------------
        self.EMW = EMW(input=frequency, unit=frequency_unit, output='cm')
        self.__wavelength = self.EMW.wavelength
        self.__frequency = self.EMW.frequency
        self.__k0 = self.EMW.k0

        # Define Roughness -----------------------------------------------------------------------------------------
        self.__roughness_unit = roughness_unit
        self.__sigma, self.__corrlength = self.__convert_roughnes(sigma, corrlength, self.__roughness_unit)

        sl = sigma / corrlength  # Roughness Parameter

        # Check for validity
        if any(sl > 0.25):
            warnings.warn("I2EM is valid for sigma/corrlength < 0.25. The actual ratio is: {0}".format(str(sl)))
        if any(self.__frequency.real > 4.5):
            warnings.warn(
                "I2EM is valid for frequency < 4.5. The actual frequency is: {0}".format(str(self.__frequency.real)))

        # Self Definitions -----------------------------------------------------------------------------------------
        self.__eps = eps
        self.__epsr = eps.real
        self.__epsi = eps.imag

        self.__corrfunc = self.__set_corrfunc(corrfunc)

        self.__n = n
        self.__kz_iza = self.__k0 * np.cos(self.iza)
        self.__kz_vza = self.__k0 * np.cos(self.vza)
        self.__phi = np.zeros_like(iza)

        # Define Static Variables for repr and str Methods ---------------------------------------------------------
        self.__vals = dict()

        self.__vals['izaDeg'] = self.izaDeg.mean()
        self.__vals['vzaDeg'] = self.vzaDeg.mean()
        self.__vals['raaDeg'] = self.raaDeg.mean()
        self.__vals['iaaDeg'] = self.iaaDeg.mean()
        self.__vals['vaaDeg'] = self.vaaDeg.mean()
        self.__vals['alphaDeg'] = self.alphaDeg.mean()
        self.__vals['betaDeg'] = self.betaDeg.mean()

        # Calculations ---------------------------------------------------------------------------------------------
        # self.__I, self.__BRF, self.__BSC = self.compute_i2em()
        self.__I = None
        self.__BRF = None
        self.__BSC = None

    # ------------------------------------------------------------------------------------------------------------------
    # Magic Methods
    # ------------------------------------------------------------------------------------------------------------------
    def __str__(self):
        self.__vals['sigma'] = self.sigma.mean()
        self.__vals['corrlength'] = self.corrlength.mean()
        self.__vals['eps'] = self.eps.mean()
        self.__vals['n'] = self.n.mean()
        self.__vals['corrfunc'] = self.corrfunc.__name__
        self.__vals['frequency'] = self.EMW.frequency.mean()
        self.__vals['wavelength'] = self.EMW.wavelength.mean()
        self.__vals['frequency_unit'] = self.EMW.frequency_unit
        self.__vals['wavelength_unit'] = self.EMW.wavelength_unit

        info = 'Class                             : I2EM\n' \
               'Mean iza and vza and raa [DEG]    : {izaDeg}, {vzaDeg}, {raaDeg}\n' \
               'Mean RMS Height                   : {sigma}\n' \
               'Mean correlation length           : {corrlength}\n' \
               'Mean dielectric constant          : {eps}\n' \
               'Correlation function              : {corrfunc}\n' \
               'Mean frequency                    : {frequency} {frequency_unit}\n' \
               'Mean wavelength                   : {wavelength} {wavelength_unit}'.format(**self.__vals)

        return info

    def __repr__(self):
        self.__vals['sigma'] = self.sigma.mean()
        self.__vals['corrlength'] = self.corrlength.mean()
        self.__vals['eps'] = self.eps.mean()
        self.__vals['n'] = self.n.mean()
        self.__vals['corrfunc'] = self.corrfunc.__name__
        self.__vals['frequency'] = self.EMW.frequency.mean()
        self.__vals['wavelength'] = self.EMW.wavelength.mean()
        self.__vals['frequency_unit'] = self.EMW.frequency_unit

        m = max(map(len, list(self.__vals.keys()))) + 1
        return '\n'.join([k.rjust(m) + ': ' + repr(v)
                          for k, v in sorted(self.__vals.items())])

    # ------------------------------------------------------------------------------------------------------------------
    # Property Calls
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def len(self):
        return len(self.iza)

    # Parameter Calls ----------------------------------------------------------------------------------------------
    @property
    def sigma(self):
        return self.__sigma

    @property
    def corrlength(self):
        return self.__corrlength

    @property
    def corrfunc(self):
        return self.__corrfunc

    @property
    def eps(self):
        return self.__epsr + self.__epsi * 1j

    @property
    def n(self):
        return self.__n.astype(np.intc)

    # Frequency Calls ----------------------------------------------------------------------------------------------
    @property
    def frequency(self):
        self.__frequence = self.EMW.frequency
        return self.__frequence

    @property
    def wavelength(self):
        self.__wavelength = self.EMW.wavelength
        return self.EMW.wavelength

    @property
    def k0(self):
        self.__k0 = self.EMW.k0
        return self.EMW.k0

    @property
    def I(self):
        if self.__I is None:
            self.__I, self.__BRF, self.__BSC = self.compute_i2em()

        return self.__I

    @property
    def BRF(self):
        if self.__BRF is None:
            self.__I, self.__BRF, self.__BSC = self.compute_i2em()

        return self.__BRF

    @property
    def BSC(self):
        if self.__BSC is None:
            self.__I, self.__BRF, self.__BSC = self.compute_i2em()

        return self.__BSC

    # ------------------------------------------------------------------------------------------------------------------
    # Property Setter
    # ------------------------------------------------------------------------------------------------------------------
    @frequency.setter
    def frequency(self, value):
        value = np.asarray(value, dtype=np.double).flatten()

        if len(value) < self.len:
            warnings.warn("The length of the input is shorter than the other parameters. The input is therefore "
                          "adjusted to the other parameters. ")

        data = (value, self.__sigma, self.__corrlength, self.__n, self.__epsr, self.__epsi)
        value, self.__sigma, self.__corrlength, self.__n, self.__epsr, self.__epsi = self.align_with(data)

        self.EMW.frequency = value
        self.__frequence = self.EMW.frequency

        self.__add_update_to_results()

    @wavelength.setter
    def wavelength(self, value):
        value = np.asarray(value, dtype=np.double).flatten()

        if len(value) < self.len:
            warnings.warn("The length of the input is shorter than the other parameters. The input is therefore "
                          "adjusted to the other parameters. ")

        data = (value, self.__sigma, self.__corrlength, self.__n, self.__epsr, self.__epsi)
        value, self.__sigma, self.__corrlength, self.__n, self.__epsr, self.__epsi = self.align_with(data)

        self.EMW.wavelength = value
        self.__wavelength = self.EMW.wavelength

        self.__add_update_to_results()

    @sigma.setter
    def sigma(self, value):
        value = np.asarray(value, dtype=np.double).flatten()

        if len(value) < self.len:
            warnings.warn("The length of the input is shorter than the other parameters. The input is therefore "
                          "adjusted to the other parameters. ")

        data = (value, self.__sigma, self.__corrlength, self.__n, self.__epsr, self.__epsi)
        value, self.__sigma, self.__corrlength, self.__n, self.__epsr, self.__epsi = self.align_with(data)

        value = self.EMW.align_with(value)

        self.__sigma = value

        self.__add_update_to_results()

    @corrlength.setter
    def corrlength(self, value):
        value = np.asarray(value, dtype=np.double).flatten()

        if len(value) < self.len:
            warnings.warn("The length of the input is shorter than the other parameters. The input is therefore "
                          "adjusted to the other parameters. ")

        data = (value, self.__sigma, self.__corrlength, self.__n, self.__epsr, self.__epsi)
        value, self.__sigma, self.__corrlength, self.__n, self.__epsr, self.__epsi = self.align_with(data)

        value = self.EMW.align_with(value)

        self.__corrlength = value

        self.__add_update_to_results()

    @corrfunc.setter
    def corrfunc(self, value):
        self.__corrfunc = self.__set_corrfunc(value)
        self.__add_update_to_results()

    @eps.setter
    def eps(self, value):
        epsr = value.real
        epsi = value.imag

        epsr = np.asarray(epsr, dtype=np.double).flatten()
        epsi = np.asarray(epsi, dtype=np.double).flatten()

        if len(value) < self.len:
            warnings.warn("The length of the input is shorter than the other parameters. The input is therefore "
                          "adjusted to the other parameters. ")

        data = (value, self.__sigma, self.__corrlength, self.__n, self.__epsr, self.__epsi)
        value, self.__sigma, self.__corrlength, self.__n, self.__epsr, self.__epsi = self.align_with(data)

        epsr, epsi = self.EMW.align_with((epsr, epsi))

        self.__epsr, self.__epsi = epsr, epsi

        self.__add_update_to_results()

    @n.setter
    def n(self, value):
        value = np.asarray(value, dtype=np.intc).flatten()

        if len(value) < self.len:
            warnings.warn("The length of the input is shorter than the other parameters. The input is therefore "
                          "adjusted to the other parameters. ")

        data = (value, self.__sigma, self.__corrlength, self.__n, self.__epsr, self.__epsi)
        value, self.__sigma, self.__corrlength, self.__n, self.__epsr, self.__epsi = self.align_with(data)

        value = self.EMW.align_with(value)

        self.__n = value

        self.__add_update_to_results()

    # -----------------------------------------------------------------------------------------------------------------
    # User callable methods
    # -----------------------------------------------------------------------------------------------------------------
    def compute_i2em(self, izaDeg=None, vzaDeg=None, raaDeg=None):
        """
        Compute the I2EM bistatic backscattering intensity.

        Parameters
        ----------
        izaDeg, vzaDeg, raaDeg: int, float or array_like
            Incidence (iza) and scattering (vza) zenith angle and relativ
            azimuth angle (ira, vra) in [DEG].

        Returns
        -------
        Intensities : SoilResult
            Intensity, BRF and BSC.

         Note
        ----
        If xzaDeg, xaaDeg, is None, the inputted angles in __init__ will be choose.

        !!! IMPORTANT !!!
        If the angles are NOT NONE, the new values will NOT be affect the property calls I, BRF and BSC!
        """

        if izaDeg is not None:
            iza = izaDeg * DEG_TO_RAD
            _, iza = align_all((self.iza, iza), dtype=np.double)
        else:
            iza = self.iza

        if vzaDeg is not None:
            vza = vzaDeg * DEG_TO_RAD
            _, vza = align_all((self.vza, vza), dtype=np.double)
        else:
            vza = self.vza

        if raaDeg is not None:
            raa = raaDeg * DEG_TO_RAD
            _, raa = align_all((self.iaa, raa), dtype=np.double)
        else:
            raa = self.raa

        kz_iza = self.__k0 * np.cos(iza)
        kz_vza = self.__k0 * np.cos(vza)

        VV, HH = i2em_wrapper(k=self.__k0, kz_iza=kz_iza, kz_vza=kz_vza, iza=iza, vza=vza,
                              raa=raa, phi=self.__phi, eps=self.__eps, corrlength=self.__corrlength,
                              sigma=self.__sigma, corrfunc=self.__corrfunc, n=self.__n)

        # Store data
        self.__I, self.__BRF, self.__BSC = self.__store(VV, HH)

        return self.__I, self.__BRF, self.__BSC

    # ------------------------------------------------------------------------------------------------------------------
    #  Auxiliary functions and private methods
    # ------------------------------------------------------------------------------------------------------------------
    def __add_update_to_results(self):
        self.__k0 = self.EMW.k0
        self.__sigma = self.sigma
        self.__corrlength = self.corrlength
        self.__n = self.n
        self.__epsr = self.eps.real
        self.__epsi = self.eps.imag
        self.__corrfunc = self.corrfunc

        self.__I, self.__BRF, self.__BSC = self.compute_i2em()

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

    def __convert_roughnes(self, sigma, corrlength, unit):
        CONVERT = {'nm': 1e9,
                   'um': 1e6,
                   'mm': 1e3,
                   'cm': 1e2,
                   'dm': 1e1,
                   'm': 1,
                   'km': 1e-3}

        if unit in CONVERT.keys():
            sigma_meter = sigma / CONVERT[unit]
            corrlength_meter = corrlength / CONVERT[unit]

            return sigma_meter * CONVERT['cm'], corrlength_meter * CONVERT['cm']

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

    class Emissivity(Angles):

        def __init__(self, iza, frequency, eps, corrlength, sigma, corrfunc='exponential', angle_unit='DEG',
                     frequency_unit='GHz'):
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

            # Align Input Parameter ------------------------------------------------------------------------------------
            vza = np.zeros_like(iza)
            raa = vza

            # iza, vza, raa, frequency, corrlength, sigma = align_all((iza, vza, raa, frequency, corrlength, sigma),
            #                                                         dtype=np.double)
            # _, eps = align_all((iza, eps), dtype=np.complex)

            eps = np.asarray(eps).flatten()
            iza, vza, raa, frequency, corrlength, sigma = asarrays((iza, vza, raa, frequency, corrlength, sigma))

            iza, vza, raa, frequency, corrlength, sigma, eps = align_all(
                (iza, vza, raa, frequency, corrlength, sigma, eps))

            super(I2EM.Emissivity, self).__init__(iza, vza, raa, normalize=False, nbar=0.0, angle_unit=angle_unit)

            self.EMW = EMW(input=frequency, unit=frequency_unit, output='cm')
            k = self.EMW.k0

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

            return EMS
