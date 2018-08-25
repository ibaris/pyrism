from __future__ import division
from pyrism.scattering.rayleigh.core.rayleigh_scatt_c import rayleigh_scattering_wrapper
from pyrism.scattering.rayleigh.core.rayleigh_phase_c import (pmatrix_wrapper, dblquad_c_wrapper, quad_c_wrapper,
                                                              dblquad_pcalc_c_wrapper, quad_pcalc_c_wrapper)
import warnings
import numpy as np
from ...core import Kernel, align_all, asarrays
import sys

# python 3.6 comparability
if sys.version_info < (3, 0):
    srange = xrange
else:
    srange = range


class Rayleigh(object):
    """
    Calculate the extinction coefficients in terms of Rayleigh
    scattering (:cite:`Ulaby.2015` and :cite:`Ulaby.2015b`).

    Parameters
    ----------
    frequency : int or float
        Frequency (GHz)
    radius : int, float or array
        Particle size a [cm].
    eps_p : complex
        Dielectric constant of the medium (a + bj).
    eps_b : complex
        Dielectric constant of the background a + bj).

    Returns
    -------
    All returns are attributes!
    self.ke : int, float or array_like
        Extinction coefficient.
    self.ks : int, float or array_like
        Scattering coefficient.
    self.ka : int, float or array_like
        Absorption coefficient.
    self.omega : int, float or array_like
        Single scattering albedo.
    self.BSC : int, float or array_like
        Backscatter coefficient sigma 0.
    """

    def __init__(self, frequency, radius, eps_p, eps_b=(1 + 1j)):

        # Check validity
        lm = 299792458 / (frequency * 1e9)  # Wavelength in meter
        self.condition = (2 * np.pi * radius) / lm

        if np.any(self.condition >= 0.5):
            warnings.warn("Rayleigh condition not holds. You should use Mie scattering.", Warning)
        else:
            pass

        radius /= 100

        frequency, radius = asarrays((frequency, radius))
        eps_p, eps_b = asarrays((eps_p, eps_b))

        frequency, radius, eps_p, eps_b = align_all((frequency, radius, eps_p, eps_b))

        self.ks = np.zeros_like(frequency, dtype=np.float)
        self.ka = np.zeros_like(frequency, dtype=np.float)
        self.kt = np.zeros_like(frequency, dtype=np.float)
        self.ke = np.zeros_like(frequency, dtype=np.float)
        self.omega = np.zeros_like(frequency, dtype=np.float)
        self.BSC = np.zeros_like(frequency, dtype=np.float)

        for i in range(frequency.shape[0]):
            self.ks[i], self.ka[i], self.kt[i], self.ke[i], self.omega[i], self.BSC[i] = rayleigh_scattering_wrapper(
                frequency[i], radius[i], eps_p[i],
                eps_b[i])

    def __str__(self):
        vals = dict()
        vals['cond'], vals['ks'], vals['ka'], vals['kt'], vals['ke'], vals[
            'omega'], vals['bsc'] = self.condition, self.ks, self.ka, self.kt, self.ke, self.omega, self.BSC

        info = 'Class                      : Rayleigh\n' \
               'Particle size              : {cond}\n' \
               'Scattering Coefficient     : {ks}\n' \
               'Absorption Coefficient     : {ka}\n' \
               'Transmission Coefficient   : {kt}\n' \
               'Extinction Coefficient     : {ke}\n' \
               'Backscattering Coefficient : {bsc}\n' \
               'Single Scattering Albedo   : {omega}'.format(**vals)

        return info

    class Phase(Kernel):
        """
        Calculate the rayleigh phase matrix (:cite:`Ulaby.2015` and :cite:`Ulaby.2015b`).

        Parameters
        ----------
        iza, vza, raa : int, float or array_like
            Incidence (iza) and scattering (vza) zenith angle, as well as relative azimuth (raa) angle.
        nbar : float, optional
            The sun or incidence zenith angle at which the isotropic term is set
            to if normalize is True. The default value is 0.0.
        angle_unit : {'DEG', 'RAD'}, optional
            * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
            * 'RAD': All input angles (iza, vza, raa) are in [RAD].

        Returns
        -------
        All returns are attributes!
        self.matrix : array_like
            Rayleigh Phase Matrix

        Notes
        -----
        You can obtain the integrated phase matrix with self.dblquad or self.quad.
        """

        def __init__(self, iza, vza, raa, normalize=False, nbar=0.0, angle_unit='DEG'):

            super(Rayleigh.Phase, self).__init__(iza=iza, vza=vza, raa=raa, normalize=normalize, nbar=nbar,
                                                 angle_unit=angle_unit,
                                                 align=True)

            self.__matrix = self.__calc()

        def __calc(self):
            if len(self.iza) > 1:
                matrix = list()
                for i in range(self.iza.shape[0]):
                    matrix.append(pmatrix_wrapper(self.iza[i], self.vza[i], self.raa[i]))

            else:
                matrix = pmatrix_wrapper(self.iza[0], self.vza[0], self.raa[0])

            return matrix

        def dblquad(self, x=[0, np.pi / 2], y=[0, 2 * np.pi], precalc=True):
            """
            Double integral of the phase matrix.

            Parameters
            ----------
            x, y : list or tuple
                X and Y are the lower (first element) and upper (second element) of the inner (theta) and outer (phi) integral,
                respectively. Default is a half-space integral with x=[0, np.pi / 2], y=[0, 2 * np.pi].
            precalc : bool
                If True and the parameter x and y are on default, the integral will be calculated with an already
                analytically solved integral. This speeds the calculation. Default is True.
            Returns
            -------
            Double integrated phase matrix : array_like
            """

            if len(x) != 2 or len(y) != 2:
                raise AssertionError(
                    "x and y must be a list or a tuple with lower bound (1. element) and upper bound (2. element)")

            a, b = y
            g, h = x

            if precalc:
                if np.allclose(a, 0) and np.allclose(b, 2 * np.pi) and np.allclose(g, 0) and np.allclose(h, np.pi / 2):

                    if len(self.iza) > 1:

                        matrix = list()
                        for i in range(self.iza.shape[0]):
                            matrix.append(dblquad_pcalc_c_wrapper(self.vza[i]))

                    else:
                        matrix = dblquad_pcalc_c_wrapper(self.vza)

            else:
                if len(self.iza) > 1:

                    matrix = list()
                    for i in range(self.iza.shape[0]):
                        matrix.append(dblquad_c_wrapper(self.vza[i], float(a), float(b), float(g), float(h)))

                else:
                    matrix = dblquad_c_wrapper(self.vza, float(a), float(b), float(g), float(h))

            self.__matrix = matrix
            # return matrix

        def quad(self, x=[0, np.pi / 2], precalc=True):
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

            if precalc:
                if np.allclose(a, 0) and np.allclose(b, np.pi / 2):

                    if len(self.iza) > 1:

                        matrix = list()
                        for i in range(self.iza.shape[0]):
                            matrix.append(quad_pcalc_c_wrapper(self.vza[i], self.raa[i]))

                    else:
                        matrix = quad_pcalc_c_wrapper(self.vza, self.raa)

            else:

                if len(self.iza) > 1:

                    matrix = list()
                    for i in range(self.iza.shape[0]):
                        matrix.append(quad_c_wrapper(self.vza[i], self.raa[i], float(a), float(b)))

                else:
                    matrix = quad_c_wrapper(self.vza[0], self.raa[0], float(a), float(b))

            self.__matrix = matrix
            # return matrix

        @property
        def matrix(self):
            return self.__matrix

        @matrix.setter
        def matrix(self, matrix):
            self.__matrix = matrix

        @property
        def HH(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][0,]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[0,])

            return item

        @property
        def VV(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][1,]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[1,])

            return item

        @property
        def VH(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][2,]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[2,])

            return item

        @property
        def HV(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][3,]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[3,])

            return item

        @property
        def p11(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][0, 0]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[0, 0])

            return item

        @property
        def p12(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][0, 1]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[0, 1])

            return item

        @property
        def p13(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][0, 2]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[0, 2])

            return item

        @property
        def p14(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][0, 3]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[0, 3])

            return item

        @property
        def p21(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][1, 0]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[1, 0])

            return item

        @property
        def p22(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][1, 1]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[1, 1])

            return item

        @property
        def p23(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][1, 2]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[1, 2])

            return item

        @property
        def p24(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][1, 3]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[1, 3])

            return item

        @property
        def p31(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][2, 0]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[2, 0])

            return item

        @property
        def p32(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][2, 1]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[2, 1])

            return item

        @property
        def p33(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][2, 2]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[2, 2])

            return item

        @property
        def p34(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][2, 3]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[2, 3])

            return item

        @property
        def p41(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][3, 0]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[3, 0])

            return item

        @property
        def p42(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][3, 1]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[3, 1])

            return item

        @property
        def p43(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][3, 2]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[3, 2])

            return item

        @property
        def p44(self):
            if len(self.iza) > 1:
                item = [np.sum(self.matrix[i][3, 3]) for i in range(self.iza.shape[0])]
                item = np.asarray(item).flatten()

            else:
                item = np.sum(self.matrix[3, 3])

            return item
