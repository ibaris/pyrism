from __future__ import division, print_function, absolute_import
import sys

import numpy as np
from respy import Angles, EM, Conversion
from respy.util import align_all, asarrays, stacks

from pyrism.core.fauxil import (reflectivity_wrapper, quad_wrapper, snell_wrapper, pol_reflection,
                                reflection_coefficients)

from pyrism.auxil import SoilResult, EPSILON, PyrismResultPol
import warnings

# python 3.6 comparability
if sys.version_info < (3, 0):
    srange = xrange
else:
    srange = range


class Fresnel(Angles):
    def __init__(self, xza, frequency, eps, sigma, angle_unit='DEG', frequency_unit='GHz', roughness_unit='cm'):
        """
        Parameters
        ----------
        xza, : int, float or ndarray
            Incidence (iza) zenith angle.
        frequency : int or float
            Frequency (GHz).
        eps : complex
            Dielectric constant of the object.
        sigma : int or float
            RMS Height. See parameter `roughness_unit`.
        angle_unit : {'DEG', 'RAD'}, optional
            * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
            * 'RAD': All input angles (iza, vza, raa) are in [RAD].
        frequency_unit : {'Hz', 'PHz', 'kHz', 'daHz', 'MHz', 'THz', 'hHz', 'GHz'}
            Unit of entered frequency. Default is 'GHz'.
        roughness_unit : {'nm', 'um', 'cm', 'dm', 'mm', 'm', 'km'}
            Unit of roughness (sigma).

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
        geometries : tuple
            If raa is defined it shows a tuple with (iza, vza, raa, alpha, beta) in [RAD]. If iaa and vaa is defined
            the tuple will be (iza, vza, iaa, vaa, alpha, beta) in [RAD]
        geometriesDeg : tuple
            If raa is defined it shows a tuple with (iza, vza, raa, alpha, beta) in [DEG]. If iaa and vaa is defined
            the tuple will be (iza, vza, iaa, vaa, alpha, beta) in [DEG]
        nbar : float
            The sun or incidence zenith angle at which the isotropic term is set
            to if normalize is True. You can change this attribute within the class.
        normlaize : bool
            Set to 'True' to make kernels 0 at nadir view illumination. Since all implemented kernels are normalized
            the default value is False.
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
        loss : array_like
            Fresnel loss term: 1 - exp(-h * cos(xza) ** 2).
        h : array_like
            Rouhness parameter: 4 * sigma ** 2 * wavenumber ** 2
        sigma : array_like
            RMS Height.
        eps : array_like
            The complex refractive index.

        matrix : array_like
            Fresnel reflectivity matrix.
        array : array_like
            Fresnel reflectivity matrix as a 4xn array where the rows of the array are the row sums of the matrix.

        I : PyrismResultPolke
            Intensity (BRDF) values for different polarisation. See pyrism.auxil.auxiliary.PyrismResultPol.
        BSC : PyrismResultPolke
            Backscattering coefficients (BSC) for different polarisation in [linear].
            See pyrism.auxil.auxiliary.PyrismResultPol.
        BSCdB : PyrismResultPolke
            Backscattering coefficients (BSC) for different polarisation in [dB].
            See pyrism.auxil.auxiliary.PyrismResultPol.

        Methods
        -------
        quad(...) : Function that returns the reflectivity matrix to use it with scipy.integrate.quad.

        See Also
        --------
        respy.Angles
        respy.EMW
        pyrism.PyrismResultPol.
        """
        # Check input parameter ------------------------------------------------------------------------------------
        vza = np.zeros_like(xza)

        # Define angles and align data -----------------------------------------------------------------------------
        eps = np.asarray(eps).flatten()

        xza, vza, frequency, sigma, epsr, epsi = align_all((xza, vza, frequency, sigma, eps.real, eps.imag))

        raa = np.zeros_like(xza)

        # NOTE: The angle xza is now self.iza
        super(Fresnel, self).__init__(iza=xza, vza=vza, raa=raa, normalize=False, angle_unit=angle_unit)

        # Define Frequency -----------------------------------------------------------------------------------------
        self.EMW = EM(frequency, frequency_unit, roughness_unit)
        self.roughness_unit = self.EMW.wavelength.unit
        self.wavelength_unit = self.EMW.wavelength.unit
        self.frequency_unit = self.EMW.frequency.unit

        # Self Definitions -----------------------------------------------------------------------------------------
        self.__pol = 4
        self.xmax = self.shape[1]

        self.__epsr, self.__epsi = epsr, epsi
        self.__eps = epsr + epsi * 1j

        self.__sigma = sigma

        # Calculations ---------------------------------------------------------------------------------------------
        self.__matrix = self.__compute_reflection_matrix()
        self.__array = self.__compute_reflection_array()
        self.__conversion = None
        self.__I, self.__BSC, self.__BSCdB = self.__add_update_to_results()

        # Define Static Variables for repr and str Methods ---------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Magic Methods
    # ------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return self.xmax

    # ------------------------------------------------------------------------------------------------------------------
    # Property Calls
    # ------------------------------------------------------------------------------------------------------------------
    # Frequency Calls ----------------------------------------------------------------------------------------------
    @property
    def frequency(self):
        return self.EMW.frequency

    @property
    def wavelength(self):
        return self.EMW.wavelength

    @property
    def wavenumber(self):
        return self.EMW.wavenumber

    # Roughness Calls ----------------------------------------------------------------------------------------------
    @property
    def loss(self):
        return 1 - np.exp(-self.h * np.cos(self.iza) ** 2)

    @property
    def h(self):
        return 4 * self.sigma ** 2 * self.wavenumber ** 2

    @property
    def sigma(self):
        return self.__sigma

    @property
    def eps(self):
        return self.__epsr + self.__epsi * 1j

    # Matrix Calls -------------------------------------------------------------------------------------------------
    @property
    def matrix(self):
        return self.__matrix

    @property
    def array(self):
        return self.__array

    # Conversion Calls ---------------------------------------------------------------------------------------------
    @property
    def I(self):
        return self.__I

    @property
    def BSC(self):
        return self.__BSC

    @property
    def BSCdB(self):
        return self.__BSCdB

    # -----------------------------------------------------------------------------------------------------------------
    # User callable methods
    # -----------------------------------------------------------------------------------------------------------------
    def quad(self, xza, eps):
        """
        Function that returns the reflectivity matrix to use it with scipy.integrate.quad.

        Parameters
        ----------
        xza : int or float
            Angle at which the function will be integrated.
        eps : complex
            Dielectric constant of the object.

        Returns
        -------
        reflectivity matrix : array_like
            Reflectivity matrix.
        """
        return reflectivity_wrapper(xza, eps)

    # ------------------------------------------------------------------------------------------------------------------
    #  Auxiliary functions and private methods
    # ------------------------------------------------------------------------------------------------------------------
    def __compute_unpolarized_part(self, array):
        U = np.zeros(array.shape[1])

        for i in srange(array.shape[1]):
            U[i] = array[array[:, i] != 0].mean()

        return U

    def __convert_bsc(self):
        self.__conversion = Conversion(value=self.__array, vza=self.iza, value_unit='BRDF', angle_unit='RAD')

    def __add_update_to_results(self):
        self.__matrix = self.__compute_reflection_matrix()
        self.__array = self.__compute_reflection_array()

        con = Conversion(value=self.__array, vza=self.iza, value_unit='BRDF', angle_unit='RAD')

        I_array = self.__array
        I_U = self.__compute_unpolarized_part(self.__array)
        I_VV = self.__array[0]
        I_HH = self.__array[1]
        I_VH = self.__array[2]
        I_HV = self.__array[3]

        I = PyrismResultPol(array=I_array,
                            U=I_U,
                            VV=I_VV,
                            HH=I_HH,
                            VH=I_VH,
                            HV=I_HV)

        BSC_array = con.BSC
        BSC_U = self.__compute_unpolarized_part(con.BSC)
        BSC_VV = con.BSC[0]
        BSC_HH = con.BSC[1]
        BSC_VH = con.BSC[2]
        BSC_HV = con.BSC[3]

        BSC = PyrismResultPol(array=BSC_array,
                              U=BSC_U,
                              VV=BSC_VV,
                              HH=BSC_HH,
                              VH=BSC_VH,
                              HV=BSC_HV)

        BSCdB_array = con.BSCdB
        BSCdB_U = self.__compute_unpolarized_part(con.BSCdB)
        BSCdB_VV = con.BSCdB[0]
        BSCdB_HH = con.BSCdB[1]
        BSCdB_VH = con.BSCdB[2]
        BSCdB_HV = con.BSCdB[3]

        BSCdB = PyrismResultPol(array=BSCdB_array,
                                U=BSCdB_U,
                                VV=BSCdB_VV,
                                HH=BSCdB_HH,
                                VH=BSCdB_VH,
                                HV=BSCdB_HV)

        return I, BSC, BSCdB

    def __compute_reflection_matrix(self):
        matrix = np.zeros((self.xmax, 4, 4))

        if self.xmax > 1:
            for i in srange(self.xmax):
                matrix[i] = reflectivity_wrapper(self.iza[i], self.eps[i]) * self.loss[i]

        else:
            matrix[0] = reflectivity_wrapper(self.iza[0], self.eps[0]) * self.loss

        return matrix

    def __compute_reflection_array(self):
        array = np.zeros((self.__pol, self.xmax))

        for i in srange(self.__pol):
            array[i] = self.__matrix[:, i].sum(axis=1)

        array[array < 0] = 0

        return array

    class Emissivity(Angles):
        def __init__(self, xza, frequency, eps, sigma, angle_unit='DEG', frequency_unit='GHz', roughness_unit='cm',
                     x=[0, np.pi / 2]):
            """
            Parameters
            ----------
            xza, : int, float or ndarray
                Incidence (iza) zenith angle.
            frequency : int or float
                Frequency (GHz).
            eps : complex
                Dielectric constant of the object.
            sigma : int or float
                RMS Height. See parameter `roughness_unit`.
            angle_unit : {'DEG', 'RAD'}, optional
                * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
                * 'RAD': All input angles (iza, vza, raa) are in [RAD].
            frequency_unit : {'Hz', 'PHz', 'kHz', 'daHz', 'MHz', 'THz', 'hHz', 'GHz'}
                Unit of entered frequency. Default is 'GHz'.
            roughness_unit : {'nm', 'um', 'cm', 'dm', 'mm', 'm', 'km'}
                Unit of roughness (sigma).
            x : list
                Integration bounds like [min, max]. Default is [0, PI/2].

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
            geometries : tuple
                If raa is defined it shows a tuple with (iza, vza, raa, alpha, beta) in [RAD]. If iaa and vaa is defined
                the tuple will be (iza, vza, iaa, vaa, alpha, beta) in [RAD]
            geometriesDeg : tuple
                If raa is defined it shows a tuple with (iza, vza, raa, alpha, beta) in [DEG]. If iaa and vaa is defined
                the tuple will be (iza, vza, iaa, vaa, alpha, beta) in [DEG]
            nbar : float
                The sun or incidence zenith angle at which the isotropic term is set
                to if normalize is True. You can change this attribute within the class.
            normlaize : bool
                Set to 'True' to make kernels 0 at nadir view illumination. Since all implemented kernels are normalized
                the default value is False.
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
            loss : array_like
                Fresnel loss term: 1 - exp(-h * cos(xza) ** 2).
            h : array_like
                Rouhness parameter: 4 * sigma ** 2 * wavenumber ** 2
            sigma : array_like
                RMS Height.
            eps : array_like
                The complex refractive index.
            x = list or tuple
                Lower and upper integration bounds as a list.
            a, b : int or float
                Lower integration (a) and upper integration (b) bound.

            matrix : array_like
                Fresnel reflectivity matrix.
            array : array_like
                Fresnel reflectivity matrix as a 4xn array where the rows of the array are the row sums of the matrix.

            Bv : PyrismResultPolke
                Intensity (BRDF) values for different polarisation. See pyrism.auxil.auxiliary.PyrismResultPol.
            BSC : PyrismResultPolke
                Backscattering coefficients (BSC) for different polarisation in [linear].
                See pyrism.auxil.auxiliary.PyrismResultPol.
            BSCdB : PyrismResultPolke
                Backscattering coefficients (BSC) for different polarisation in [dB].
                See pyrism.auxil.auxiliary.PyrismResultPol.

            Methods
            -------
            quad(...) : Function that returns the reflectivity matrix to use it with scipy.integrate.quad.

            See Also
            --------
            respy.Angles
            respy.EMW
            pyrism.PyrismResultPol.
            """
            # Check input parameter ------------------------------------------------------------------------------------
            vza = np.zeros_like(xza)

            # Define angles and align data -----------------------------------------------------------------------------
            eps = np.asarray(eps).flatten()

            xza, vza, frequency, sigma, epsr, epsi = align_all((xza, vza, frequency, sigma, eps.real, eps.imag))

            raa = np.zeros_like(xza)

            # NOTE: The angle xza is now self.iza
            super(Fresnel.Emissivity, self).__init__(iza=xza, vza=vza, raa=raa, normalize=False, angle_unit=angle_unit)

            # Define Frequency -----------------------------------------------------------------------------------------
            self.EMW = EMW(frequency, frequency_unit, roughness_unit)
            self.roughness_unit = self.EMW.wavelength_unit
            self.wavelength_unit = self.EMW.wavelength_unit
            self.frequency_unit = self.EMW.frequency_unit

            # Self Definitions -----------------------------------------------------------------------------------------
            self.__pol = 4
            self.xmax = self.shape[1]

            self.__epsr, self.__epsi = epsr, epsi
            self.__eps = epsr + epsi * 1j

            self.__sigma = sigma
            self.__x = x
            self.__a = x[0]
            self.__b = x[1]

            # Calculations ---------------------------------------------------------------------------------------------
            self.__matrix = self.quad(a=self.__a, b=self.__b)
            self.__array = self.__compute_emissivity_array()
            self.__Bv = self.__add_update_to_results()

            # Define Static Variables for repr and str Methods ---------------------------------------------------------
            self.__vals = dict()
            self.__vals['xza'] = self.izaDeg.mean()
            self.__vals['roughness_unit'] = self.roughness_unit
            self.__vals['frequency_unit'] = self.EMW.frequency_unit
            self.__vals['wavelength_unit'] = self.EMW.wavelength_unit
            self.__vals['normalize'] = self.normalize
            self.__vals['nbar'] = self.nbar
            self.__vals['angle_unit'] = self.angle_unit
            self.vals = self.__vals

            # ------------------------------------------------------------------------------------------------------------------
            # Magic Methods
            # ------------------------------------------------------------------------------------------------------------------

        def __str__(self):
            self.__vals['sigma'] = self.sigma.mean()
            self.__vals['frequency'] = self.EMW.frequency.mean()
            self.__vals['wavelength'] = self.EMW.wavelength.mean()
            self.__vals['eps'] = self.eps
            self.__vals['loss'] = self.loss.mean()

            info = 'Class                      : Fresnel.Emissivity\n' \
                   'Mean zenith angle [DEG]    : {xza} \n' \
                   'Mean sigma                 : {sigma} {roughness_unit}\n' \
                   'Mean loss                  : {loss}\n' \
                   'Mean frequency             : {frequency} {frequency_unit}\n' \
                   'Mean wavelength            : {wavelength} {wavelength_unit}'.format(**self.__vals)

            return info

        def __repr__(self):
            self.__vals['sigma'] = self.sigma.mean()
            self.__vals['frequency'] = self.EMW.frequency.mean()
            self.__vals['wavelength'] = self.EMW.wavelength.mean()
            self.__vals['eps'] = self.eps

            m = max(map(len, list(self.__vals.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.__vals.items())])

        def __len__(self):
            return self.xmax

        # --------------------------------------------------------------------------------------------------------------
        # Property Calls
        # --------------------------------------------------------------------------------------------------------------
        @property
        def a(self):
            return self.__a

        @property
        def b(self):
            return self.__b

        @property
        def x(self):
            return self.__x

        # Frequency Calls ------------------------------------------------------------------------------------------
        @property
        def frequency(self):
            return self.EMW.frequency

        @property
        def wavelength(self):
            return self.EMW.wavelength

        @property
        def wavenumber(self):
            return self.EMW.wavenumber

        # Roughness Calls ------------------------------------------------------------------------------------------
        @property
        def loss(self):
            return 1 - np.exp(-self.h * np.cos(self.iza) ** 2)

        @property
        def h(self):
            return 4 * self.sigma ** 2 * self.wavenumber ** 2

        @property
        def sigma(self):
            return self.__sigma

        @property
        def eps(self):
            return self.__epsr + self.__epsi * 1j

        # Matrix Calls ---------------------------------------------------------------------------------------------
        @property
        def matrix(self):
            return self.__matrix

        @property
        def array(self):
            return self.__array

        # Conversion Calls -----------------------------------------------------------------------------------------
        @property
        def Bv(self):
            return self.__Bv

        # ------------------------------------------------------------------------------------------------------------------
        # Property Setter
        # ------------------------------------------------------------------------------------------------------------------
        @a.setter
        def a(self, value):
            if isinstance(value, (float, int)):
                self.__a = value
                self.__Bv = self.__add_update_to_results()
            else:
                raise ValueError("Lower bound must be int or float.")

        @b.setter
        def b(self, value):
            if isinstance(value, (float, int)):
                self.__b = value
                self.__Bv = self.__add_update_to_results()
            else:
                raise ValueError("Upper bound must be int or float.")

        @x.setter
        def x(self, value):
            if isinstance(value, (list, tuple)):
                if isinstance(value[0], (float, int)):
                    self.__a = value
                else:
                    raise ValueError("Lower bound must be int or float.")

                if isinstance(value[1], (float, int)):
                    self.__b = value
                else:
                    raise ValueError("Upper bound must be int or float.")

                self.__Bv = self.__add_update_to_results()

            else:
                raise ValueError("Value must be a tuple or a list.")

        # Frequency Setter ---------------------------------------------------------------------------------------------
        @frequency.setter
        def frequency(self, value):
            value = np.asarray(value, dtype=np.double).flatten()

            if len(value) < self.len:
                warnings.warn("The length of the input is shorter than the other parameters. The input is therefore "
                              "adjusted to the other parameters. ")

            data = (value, self.__sigma, self.__epsr, self.__epsi)
            value, self.__sigma, self.__epsr, self.__epsi = self.align_with(data)

            self.EMW.frequency = value

            self.__Bv = self.__add_update_to_results()

        @wavelength.setter
        def wavelength(self, value):
            value = np.asarray(value, dtype=np.double).flatten()

            if len(value) < self.len:
                warnings.warn("The length of the input is shorter than the other parameters. The input is therefore "
                              "adjusted to the other parameters. ")

            data = (value, self.__sigma, self.__epsr, self.__epsi)
            value, self.__sigma, self.__epsr, self.__epsi = self.align_with(data)

            self.EMW.wavelength = value

            self.__Bv = self.__add_update_to_results()

        # Roughness Setter ---------------------------------------------------------------------------------------------
        @sigma.setter
        def sigma(self, value):
            value = np.asarray(value, dtype=np.double).flatten()

            if len(value) < self.len:
                warnings.warn("The length of the input is shorter than the other parameters. The input is therefore "
                              "adjusted to the other parameters. ")

            data = (value, self.__sigma, self.__epsr, self.__epsi)
            value, self.__sigma, self.__epsr, self.__epsi = self.align_with(data)

            value = self.EMW.align_with(value)

            self.__sigma = value.flatten()

            self.__Bv = self.__add_update_to_results()

        @eps.setter
        def eps(self, value):
            value = np.asarray(value).flatten()

            epsr = value.real
            epsi = value.imag

            epsr = np.asarray(epsr, dtype=np.double).flatten()
            epsi = np.asarray(epsi, dtype=np.double).flatten()

            if len(epsr) < self.len:
                warnings.warn("The length of the input is shorter than the other parameters. The input is therefore "
                              "adjusted to the other parameters. ")

            data = (epsr, epsi, self.__sigma, self.__epsr, self.__epsi)
            epsr, epsi, self.__sigma, self.__epsr, self.__epsi = self.align_with(data)

            epsr, epsi = self.EMW.align_with((epsr, epsi))

            self.__epsr, self.__epsi = epsr, epsi

            self.__Bv = self.__add_update_to_results()

        # --------------------------------------------------------------------------------------------------------------
        # User callable methods
        # --------------------------------------------------------------------------------------------------------------
        def quad(self, a=0, b=np.pi / 2):
            """
            Integral of the phase matrix with neglecting the phi dependence.

            Parameters
            ----------
            a : int or float
                Lower integration bound. Default is 0.
            b : int or float
                Upper integration bound. Default is PI/2.

            Returns
            -------
            Integrated phase matrix : array_like
            """
            matrix = np.zeros((self.xmax, 4, 4))

            if self.xmax > 1:
                for i in srange(self.xmax):
                    matrix[i] = quad_wrapper(float(a), float(b), self.eps[i])

            else:
                matrix[0] = quad_wrapper(float(a), float(b), self.eps[0])

            return matrix

        # --------------------------------------------------------------------------------------------------------------
        #  Auxiliary functions and private methods
        # --------------------------------------------------------------------------------------------------------------
        def __compute_emissivity_array(self):
            array = np.zeros((self.__pol, self.xmax))

            for i in srange(self.__pol):
                array[i] = self.__matrix[:, i].sum(axis=1)

            array[array < 0] = 0

            return array

        def __add_update_to_results(self):
            self.__matrix = self.quad(a=self.__a, b=self.__b)
            self.__array = self.__compute_emissivity_array()

            Bv_array = self.__array
            Bv_U = self.__compute_unpolarized_part(self.__array)
            Bv_VV = self.__array[0]
            Bv_HH = self.__array[1]
            Bv_VH = self.__array[2]
            Bv_HV = self.__array[3]

            Bv = PyrismResultPol(array=Bv_array,
                                 U=Bv_U,
                                 VV=Bv_VV,
                                 HH=Bv_HH,
                                 VH=Bv_VH,
                                 HV=Bv_HV)

            return Bv

        def __compute_unpolarized_part(self, array):
            U = np.zeros(array.shape[1])

            for i in srange(array.shape[1]):
                U[i] = array[array[:, i] != 0].mean()

            return U
