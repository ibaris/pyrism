from __future__ import division, print_function, absolute_import

import sys

import numpy as np
from radarpy import Angles, align_all, asarrays, BSC, BRF, dB, stacks
from ...core.fauxil import reflectivity_wrapper, transmissivity_wrapper, quad_wrapper, snell_wrapper

from ...auxil import SoilResult

EPSILON = sys.float_info.epsilon  # typical floating-point calculation error


class Fresnel(Angles):
    def __init__(self, iza, frequency, n1, n2, sigma, normalize=False, nbar=0.0, angle_unit='DEG',
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
        frequency, sigma = asarrays((frequency, sigma))
        n1, n2 = asarrays((n1, n2))

        self.iso = 1

        self.frequency = frequency
        self.k0 = 2 * np.pi * frequency / 30
        self.n1 = n1
        self.n2 = n2
        self.sigma = sigma

        raa = np.zeros_like(iza)
        vza = np.zeros_like(iza)

        super(Fresnel, self).__init__(iza, vza, raa, normalize, nbar, angle_unit)

        iza, frequency, sigma, self.n1, self.n2 = align_all((self.iza, frequency, sigma, n1, n2))

        self.frequency = frequency.real
        self.sigma = sigma.real
        self.iza = iza.real
        self.rza = self.__rza_calc()

        self.h = 4 * self.sigma ** 2 * self.k0 ** 2
        self.loss = np.exp(-self.h * np.cos(self.iza) ** 2)

        self.__rmatrix = self.__rcalc()
        self.__tmatrix = self.__tcalc()
        self.__ematrix = self.__ecalc(x)

        self.__store()

    def __rza_calc(self):
        if len(self.iza) > 1:
            matrix = list()
            for i in range(self.iza.shape[0]):
                matrix.append(snell_wrapper(self.iza[i], self.n1[i], self.n2[i]))

        else:
            matrix = snell_wrapper(self.iza[0], self.n1[0], self.n2[0])

        return np.asarray(matrix)

    def __rcalc(self):
        if len(self.iza) > 1:
            matrix = list()
            for i in range(self.iza.shape[0]):
                matrix.append(reflectivity_wrapper(self.iza[i], self.n1[i], self.n2[i], self.iso) * self.loss[i])

        else:
            matrix = reflectivity_wrapper(self.iza[0], self.n1[0], self.n2[0], self.iso) * self.loss

        return matrix

    def __tcalc(self):
        if len(self.iza) > 1:
            matrix = list()
            for i in range(self.iza.shape[0]):
                matrix.append(transmissivity_wrapper(self.iza[i], self.n1[i], self.n2[i], self.iso))

        else:
            matrix = transmissivity_wrapper(self.iza[0], self.n1[0], self.n2[0], self.iso)

        return matrix

    def __ecalc(self, x):
        matrix = self.quad(x)

        if len(self.iza) > 1:
            matrix = [1 - matrix[i] for i in range(self.iza.shape[0])]

        else:
            matrix = 1 - matrix

        return matrix

    def quad(self, x=[0, np.pi / 2]):
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
        if len(x) != 2:
            raise AssertionError(
                "x must be a list or a tuple with lower bound (1. element) and upper bound (2. element)")

        a, b = x

        if len(self.iza) > 1:

            matrix = list()
            for i in range(self.iza.shape[0]):
                matrix.append(quad_wrapper(float(a), float(b), self.n1[i], self.n2[i], self.iso))

        else:
            matrix = quad_wrapper(float(a), float(b), self.n1[0], self.n2[0], self.iso)

        return matrix

    def __store(self):

        self.I = SoilResult(ISO=self.__pol(self.__rmatrix, 'iso'),
                            VV=self.__pol(self.__rmatrix, 'VV'),
                            HH=self.__pol(self.__rmatrix, 'HH'),
                            VH=self.__pol(self.__rmatrix, 'VH'),
                            HV=self.__pol(self.__rmatrix, 'HV'))

        self.I['array'] = stacks((self.I.ISO, self.I.VV, self.I.HH, self.I.VH, self.I.HV))

        self.BSC = SoilResult(ISO=BSC(self.I.ISO, self.vza),
                              VV=BSC(self.I.VV, self.vza),
                              HH=BSC(self.I.HH, self.vza),
                              VH=BSC(self.I.VH, self.vza),
                              HV=BSC(self.I.HV, self.vza),

                              ISOdB=dB(BSC(self.I.ISO, self.vza)),
                              VVdB=dB(BSC(self.I.VV, self.vza)),
                              HHdB=dB(BSC(self.I.HH, self.vza)),
                              VHdB=dB(BSC(self.I.VH, self.vza)),
                              HVdB=dB(BSC(self.I.HV, self.vza)))

        self.BSC['array'] = stacks((self.BSC.ISO, self.BSC.VV, self.BSC.HH, self.BSC.VH, self.BSC.HV))
        self.BSC['arraydB'] = stacks((self.BSC.ISOdB, self.BSC.VVdB, self.BSC.HHdB, self.BSC.VHdB, self.BSC.HVdB))

        self.BRF = SoilResult(ISO=BRF(self.I.ISO),
                              VV=BRF(self.I.VV),
                              HH=BRF(self.I.HH),
                              VH=BRF(self.I.VH),
                              HV=BRF(self.I.HV))

        self.BRF['array'] = stacks((self.BRF.ISO, self.BRF.VV, self.BRF.HH, self.BRF.VH, self.BRF.HV))

        self.E = SoilResult(ISO=self.__pol(self.__ematrix, 'iso'),
                            VV=self.__pol(self.__ematrix, 'VV'),
                            HH=self.__pol(self.__ematrix, 'HH'),
                            VH=self.__pol(self.__ematrix, 'VH'),
                            HV=self.__pol(self.__ematrix, 'HV'))

        self.E['array'] = stacks((self.E.ISO, self.E.VV, self.E.HH, self.E.VH, self.E.HV))

    def __pol(self, input, pol):
        if self.iso == 0:
            if pol == 'VV':
                selection = [0, ]

            if pol == 'HH':
                selection = [1, ]

            if pol == 'VH':
                selection = [2, ]

            if pol == 'HV':
                selection = [3, ]

        else:
            if pol == 'iso':
                selection = [0, ]

            if pol == 'VV':
                selection = [1, ]

            if pol == 'HH':
                selection = [2, ]

            if pol == 'VH':
                selection = [3, ]

            if pol == 'HV':
                selection = [4, ]

        if len(self.iza) > 1:
            if self.iso == 0 and pol == 'iso':
                return np.zeros_like(self.iza)

            else:
                result = [np.sum(input[i][selection]) for i in range(self.iza.shape[0])]
                return np.asarray(result)

        else:
            if self.iso == 0 and pol == 'iso':
                return np.zeros_like(self.iza)

            else:
                return np.sum(input[selection])

    @property
    def rmatrix(self):
        return self.__rmatrix

    @property
    def ematrix(self):
        return self.__ematrix
