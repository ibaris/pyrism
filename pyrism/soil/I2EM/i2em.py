# -*- coding: utf-8 -*-
from __future__ import division

import warnings
from respy import Angles, EM, Quantity, Units, Conversion

import numpy as np
from pyrism.auxil import SoilResult, EmissivityResult, PyrismResult
from pyrism.cython_iem import i2em, ixx, i2em_ems
from pyrism.soil.I2EM.core import (exponential, gaussian, xpower, mixed)
from respy.constants import deg_to_rad
from respy.util import align_all, asarrays


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
        wavenumber : array_like
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
        self.EM = EM(input=frequency, unit=frequency_unit, output='cm')
        self.__wavelength = self.EM.wavelength
        self.__frequency = self.EM.frequency
        self.__wavenumber = self.EM.wavenumber

        # Define Roughness -----------------------------------------------------------------------------------------
        self.__sigma = Quantity(sigma, roughness_unit)
        self.__corrlength = Quantity(corrlength, roughness_unit)

        if self.__sigma.unit != Units.length.cm:
            self.__sigma = self.__sigma.convert_to('cm')
            self.__corrlength = self.__corrlength.convert_to('cm')

        sl = self.__sigma / self.__corrlength  # Roughness Parameter

        # Check for validity
        if any(sl > 0.25):
            warnings.warn(
                "I2EM is valid for sigma/corrlength < 0.25. The actual ratio is: {0}".format(str(sl[sl > 0.25])))

        if any(self.__frequency.real > 4.5):
            warnings.warn(
                "I2EM is valid for frequency < 4.5. The actual frequency is: "
                "{0}".format(str(self.__frequency[self.__frequency > 4.5])))

        # Self Definitions -----------------------------------------------------------------------------------------
        self.__eps = eps

        self.__corrfunc = self.__set_corrfunc(corrfunc)

        self.__n = n
        self.__kz_iza = self.__wavenumber * np.cos(self.iza)
        self.__kz_vza = self.__wavenumber * np.cos(self.vza)
        self.__phi = np.zeros_like(iza)

        # Calculations ---------------------------------------------------------------------------------------------
        # self.__I, self.__BRF, self.__BSC = self.compute_i2em()
        self.__I = None
        self.__BRF = None
        self.__BSC = None
        self.__conversion = None

    # ------------------------------------------------------------------------------------------------------------------
    # Magic Methods
    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self):
        return len(self.iza)

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
        return Quantity(self.eps, name="Relative Permittivity of Soil", constant=True)

    @property
    def n(self):
        return self.__n.astype(np.intc)

    # Frequency Calls ----------------------------------------------------------------------------------------------
    @property
    def frequency(self):
        self.__frequence = self.EM.frequency
        return self.__frequence

    @property
    def wavelength(self):
        self.__wavelength = self.EM.wavelength
        return self.EM.wavelength

    @property
    def wavenumber(self):
        self.__wavenumber = self.EM.wavenumber
        return self.EM.wavenumber

    @property
    def I(self):
        if self.__I is None:
            self.__convert_BSC()

        return self.__I

    @property
    def BRF(self):
        if self.__BRF is None:
            self.__convert_BSC()

        return self.__BRF

    @property
    def BSC(self):
        if self.__BSC is None:
            self.__convert_BSC()

        return self.__BSC

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
            iza = izaDeg * deg_to_rad
            _, iza = align_all((self.iza, iza), dtype=np.double)
        else:
            iza = self.iza

        if vzaDeg is not None:
            vza = vzaDeg * deg_to_rad
            _, vza = align_all((self.vza, vza), dtype=np.double)
        else:
            vza = self.vza

        if raaDeg is not None:
            raa = raaDeg * deg_to_rad
            _, raa = align_all((self.iaa, raa), dtype=np.double)
        else:
            raa = self.raa

        angles = Angles(iza=iza, vza=vza, raa=raa, angle_unit='RAD')

        kz_iza = self.__wavenumber * np.cos(angles.iza)
        kz_vza = self.__wavenumber * np.cos(angles.vza)

        np.warnings.filterwarnings('ignore')

        VV, HH = i2em(k=self.__wavenumber.value, kz_iza=kz_iza.value, kz_vza=kz_vza.value,
                      iza=angles.iza.value, vza=angles.vza.value, raa=angles.raa.value,
                      phi=self.__phi, eps=self.__eps, corrlength=self.__corrlength.value,
                      sigma=self.__sigma.value, corrfunc=self.__corrfunc, n=self.__n)

        self.Ivv, self.Ihh = ixx(k=self.__wavenumber.value, kz_iza=kz_iza.value, kz_vza=kz_vza.value,
                                 iza=angles.iza.value, vza=angles.vza.value, raa=angles.raa.value,
                                 phi=self.__phi, eps=self.__eps, corrlength=self.__corrlength.value,
                                 sigma=self.__sigma.value, corrfunc=self.__corrfunc, n=self.__n)

        result = np.array([VV, HH])

        return Conversion(result, angles.vza.value, 'BSC')

    # ------------------------------------------------------------------------------------------------------------------
    #  Auxiliary functions and private methods
    # ------------------------------------------------------------------------------------------------------------------

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

    def __convert_BSC(self):
        if self.__conversion is None:
            self.__conversion = self.compute_i2em()

        self.c = self.__conversion

        self.__I = PyrismResult(array=self.__conversion.I.transpose(),
                                VV=self.__conversion.I[0],
                                HH=self.__conversion.I[1])

        self.__BRF = PyrismResult(array=self.__conversion.BRF.transpose(),
                                  VV=self.__conversion.BRF[0],
                                  HH=self.__conversion.BRF[1])

        self.__BSC = PyrismResult(array=self.__conversion.BSC.transpose(),
                                  arraydB=self.__conversion.BSCdB.transpose(),
                                  VV=self.__conversion.BSC[0],
                                  HH=self.__conversion.BSC[1],
                                  VVdB=self.__conversion.BSCdB[0],
                                  HHdB=self.__conversion.BSCdB[1])

        # self.__I.array.set_name('Intensity [[VV], [HH]]')
        self.__I.VV.set_name('Intensity (VV)')
        self.__I.HH.set_name('Intensity (HH)')
        self.__I.array.set_name('Intensity (VV, HH)')

        # self.__BRF.array.set_name('Bidirectional Reflectance Factor [[VV], [HH]]')
        self.__BRF.VV.set_name('Bidirectional Reflectance Factor (VV)')
        self.__BRF.VV.set_constant(True)
        self.__BRF.HH.set_name('Bidirectional Reflectance Factor (HH)')
        self.__BRF.HH.set_constant(True)
        self.__BRF.array.set_name('Bidirectional Reflectance Factor (VV, HH)')
        self.__BRF.array.set_constant(True)

        # self.__BSC.array.set_name('Backscattering Coefficient [[VV], [HH]]')
        self.__BSC.VV.set_name('Backscattering Coefficient (VV)')
        self.__BSC.VV.set_constant(True)
        self.__BSC.HH.set_name('Backscattering Coefficient (HH)')
        self.__BSC.HH.set_constant(True)
        self.__BSC.array.set_name('Backscattering Coefficient (VV, HH)')
        self.__BSC.array.set_constant(True)

        self.__BSC.VVdB.set_name('Backscattering Coefficient (VV)')
        self.__BSC.VVdB.set_constant(True)
        self.__BSC.HHdB.set_name('Backscattering Coefficient (HH)')
        self.__BSC.HHdB.set_constant(True)
        self.__BSC.arraydB.set_name('Backscattering Coefficient (VV, HH)')
        self.__BSC.arraydB.set_constant(True)

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

            self.EM = EM(input=frequency, unit=frequency_unit, output='cm')
            k = self.EM.wavenumber

            corrfunc = self.__set_corrfunc(corrfunc)

            # calculate I2EM BSC
            V, H = i2em_ems(self.iza, k, sigma, corrlength, eps, corrfunc)

            self.EMS = self.__convert_BSC(V, H)

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

        def __convert_BSC(self, V, H):

            EMS = EmissivityResult(array=np.array([[V], [H]]),
                                   arraydB=np.array([[dB(V)], [dB(H)]]),
                                   V=V,
                                   H=H,
                                   VdB=dB(V),
                                   HdB=dB(H))

            return EMS
