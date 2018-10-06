from __future__ import division, print_function, absolute_import

import sys

import numpy as np
from radarpy import Angles, align_all, asarrays, BSC, BRF, dB, stacks, BRDF
from pyrism.core.fauxil import reflectivity_wrapper, quad_wrapper, snell_wrapper

from ...auxil import SoilResult

EPSILON = sys.float_info.epsilon  # typical floating-point calculation error


class Fresnel(Angles):
    def __init__(self, xza, frequency, eps, sigma, normalize=False, nbar=0.0, angle_unit='DEG'):
        """
        Parameters
        ----------
        iza, : int, float or ndarray
            Incidence (iza) zenith angle.
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
            Frequency (GHz).
        n1, n2 : complex
            Complex refractive indices of incident (n1) and underlying (n2) medium.
        sigma : int or float
            RMS Height (cm)
        x : list
            Integration lower and upper bound as a list to calculate the emissivity. Default ist x=[0, pi/2].
        """
        xza, frequency, sigma = asarrays((xza, frequency, sigma))

        eps = np.asarray(eps).flatten()

        xza, frequency, sigma = align_all((xza, frequency, sigma))

        _, eps = align_all((xza, eps))

        raa = np.zeros_like(xza)
        vza = np.zeros_like(xza)

        super(Fresnel, self).__init__(iza=xza, vza=vza, raa=raa, normalize=normalize, angle_unit=angle_unit)

        self.xza = self.iza

        self.frequency = frequency
        self.k0 = 2 * np.pi * frequency / 30
        self.eps = eps
        self.sigma = sigma

        self.k0 = 2 * np.pi * frequency / 30
        self.h = 4 * self.sigma ** 2 * self.k0 ** 2
        self.loss = np.exp(-self.h * np.cos(self.xza) ** 2)

        self.__rmatrix = self.__rcalc()
        self.I, self.BSC, self.BRF = self.__store(self.VV, self.HH, self.VH, self.HV)

    @property
    def rmatrix(self):
        return self.__rmatrix

    @property
    def VV(self):
        if isinstance(self.__rmatrix, list):
            pol = np.zeros((1, len(self.__rmatrix))).flatten()
            for i, item in enumerate(self.__rmatrix):
                pol[i] = item[0].sum() * self.loss[i]

        else:
            pol = self.__rmatrix[0].sum() * self.loss

        return pol

    @property
    def HH(self):
        if isinstance(self.__rmatrix, list):
            pol = np.zeros((1, len(self.__rmatrix))).flatten()
            for i, item in enumerate(self.__rmatrix):
                pol[i] = item[1].sum() * self.loss[i]

        else:
            pol = self.__rmatrix[1].sum() * self.loss

        return pol

    @property
    def VH(self):
        if isinstance(self.__rmatrix, list):
            pol = np.zeros((1, len(self.__rmatrix))).flatten()
            for i, item in enumerate(self.__rmatrix):
                pol[i] = item[2].sum() * self.loss[i]

        else:
            pol = self.__rmatrix[2].sum() * self.loss

        return pol

    @property
    def HV(self):
        if isinstance(self.__rmatrix, list):
            pol = np.zeros((1, len(self.__rmatrix))).flatten()
            for i, item in enumerate(self.__rmatrix):
                pol[i] = item[3].sum() * self.loss[i]

        else:
            pol = self.__rmatrix[3].sum() * self.loss

        return pol

    def __rcalc(self):
        if len(self.xza) > 1:
            matrix = list()
            for i in range(self.xza.shape[0]):
                matrix.append(reflectivity_wrapper(self.xza[i], self.eps[i]))

        else:
            matrix = reflectivity_wrapper(self.xza[0], self.eps[0])

        return matrix

    def __store(self, VV, HH, VH, HV):

        I = SoilResult(array=np.array([VV, HH, VH, HV]),
                       VV=VV,
                       HH=HH,
                       VH=VH,
                       HV=HV)

        BSC_ = SoilResult(array=np.array([[BSC(VV, self.vza)],
                                          [BSC(HH, self.vza)],
                                          [BSC(VH, self.vza)],
                                          [BSC(HV, self.vza)]]),

                          arraydB=np.array([[dB(BSC(VV, self.vza))],
                                            [dB(BSC(HH, self.vza))],
                                            [dB(BSC(VH, self.vza))],
                                            [dB(BSC(HV, self.vza))]]),

                          VV=BSC(VV, self.vza),
                          HH=BSC(HH, self.vza),
                          VH=BSC(VH, self.vza),
                          HV=BSC(HV, self.vza),
                          VVdB=dB(BSC(VV, self.vza)),
                          HHdB=dB(BSC(HH, self.vza)),
                          VHdB=dB(BSC(VH, self.vza)),
                          HVdB=dB(BSC(HV, self.vza)))

        BRF_ = SoilResult(array=np.array([[BRF(I.VV)],
                                          [BRF(I.HH)],
                                          [BRF(I.VH)],
                                          [BRF(I.HV)]]),

                          VV=BRF(I.VV),
                          HH=BRF(I.HH),
                          VH=BRF(I.VH),
                          HV=BRF(I.HV))

        return I, BRF_, BSC_

    class Emissivity(Angles):
        def __init__(self, xza, frequency, eps, sigma, normalize=False, nbar=0.0, angle_unit='DEG',
                     x=[0, np.pi / 2]):
            """
            Parameters
            ----------
            iza, : int, float or ndarray
                Incidence (iza) zenith angle.
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
                Frequency (GHz).
            n1, n2 : complex
                Complex refractive indices of incident (n1) and underlying (n2) medium.
            sigma : int or float
                RMS Height (cm)
            x : list
                Integration lower and upper bound as a list to calculate the emissivity. Default ist x=[0, pi/2].
            """
            if len(x) != 2:
                raise AssertionError(
                    "x must be a list or a tuple with lower bound (1. element) and upper bound (2. element)")

            xza, frequency, sigma = asarrays((xza, frequency, sigma))

            eps = np.asarray(eps).flatten()

            xza, frequency, sigma = align_all((xza, frequency, sigma))

            _, eps = align_all((xza, eps))

            raa = np.zeros_like(xza)
            vza = np.zeros_like(xza)

            super(Fresnel.Emissivity, self).__init__(iza=xza, vza=vza, raa=raa, normalize=normalize,
                                                     angle_unit=angle_unit)

            self.xza = self.iza

            self.a = x[0]
            self.b = x[1]

            self.frequency = frequency
            self.k0 = 2 * np.pi * frequency / 30
            self.eps = eps
            self.sigma = sigma

            self.k0 = 2 * np.pi * frequency / 30
            self.h = 4 * self.sigma ** 2 * self.k0 ** 2
            self.loss = np.exp(-self.h * np.cos(self.xza) ** 2)

            self.__ematrix = self.__ecalc()
            self.EMS = self.__store(self.VV, self.HH, self.VH, self.HV)

        def __quad(self, a=0, b=np.pi / 2):
            """
            Integral of the phase matrix with neglecting the phi dependence.

            Parameters
            ----------
            x : list or tuple
                X are the lower (first element) and upper (second element) of the integral (theta). Default is a
                half-space integral with x=[0, np.pi / 2].
            precalc : bool
                If True and the parameter x is on default, the integral will be calculated with an already
                analytically solved integral. This speeds the calculation. Default is True.

            Returns
            -------
            Integrated phase matrix : array_like
            """
            if len(self.xza) > 1:

                matrix = list()
                for i in range(self.xza.shape[0]):
                    matrix.append(quad_wrapper(float(a), float(b), self.eps[i]))

            else:
                matrix = quad_wrapper(float(a), float(b), self.eps[0])

            return matrix

        def __ecalc(self):
            if len(self.xza) > 1:
                matrix = list()
                for i in range(self.xza.shape[0]):
                    matrix.append(self.__quad(self.a, self.b))

            else:
                matrix = self.__quad(self.a, self.b)

            return matrix

        @property
        def rmatrix(self):
            return self.__ematrix

        @property
        def VV(self):
            if isinstance(self.__ematrix, list):
                pol = np.zeros((1, len(self.__ematrix))).flatten()
                for i, item in enumerate(self.__ematrix):
                    pol[i] = item[0].sum() * self.loss[i]

            else:
                pol = self.__ematrix[0].sum() * self.loss

            return pol

        @property
        def HH(self):
            if isinstance(self.__ematrix, list):
                pol = np.zeros((1, len(self.__ematrix))).flatten()
                for i, item in enumerate(self.__ematrix):
                    pol[i] = item[1].sum() * self.loss[i]

            else:
                pol = self.__ematrix[1].sum() * self.loss

            return pol

        @property
        def VH(self):
            if isinstance(self.__ematrix, list):
                pol = np.zeros((1, len(self.__ematrix))).flatten()
                for i, item in enumerate(self.__ematrix):
                    pol[i] = item[2].sum() * self.loss[i]

            else:
                pol = self.__ematrix[2].sum() * self.loss

            return pol

        @property
        def HV(self):
            if isinstance(self.__ematrix, list):
                pol = np.zeros((1, len(self.__ematrix))).flatten()
                for i, item in enumerate(self.__ematrix):
                    pol[i] = item[3].sum() * self.loss[i]

            else:
                pol = self.__ematrix[3].sum() * self.loss

            return pol

        def __store(self, VV, HH, VH, HV):

            EMS = SoilResult(array=np.array([VV, HH, VH, HV]),
                             VV=VV,
                             HH=HH,
                             VH=VH,
                             HV=HV)

            return EMS
