from __future__ import division, print_function, absolute_import

from .core.auxiliary import reflectivity_wrapper, transmissivity_wrapper, quad_wrapper, snell_wrapper
from ...core import SoilResult

from radarpy import Angles, align_all, asarrays

import sys

import numpy as np
import scipy as sp
from numpy import cos, pi

EPSILON = sys.float_info.epsilon  # typical floating-point calculation error


class Fresnel(Angles):
    def __init__(self, iza, frequency, n1, n2, sigma, normalize=False, nbar=0.0, angle_unit='DEG', isometric=False):

        frequency, sigma = asarrays((frequency, sigma))
        n1, n2 = asarrays((n1, n2))

        self.iso = 0 if not isometric else 1

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

        self.__rmatrix = self.__rcalc()
        self.__tmatrix = self.__tcalc()

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
                matrix.append(reflectivity_wrapper(self.iza[i], self.n1[i], self.n2[i], self.iso))

        else:
            matrix = reflectivity_wrapper(self.iza[0], self.n1[0], self.n2[0], self.iso)

        return matrix

    def __tcalc(self):
        if len(self.iza) > 1:
            matrix = list()
            for i in range(self.iza.shape[0]):
                matrix.append(transmissivity_wrapper(self.iza[i], self.n1[i], self.n2[i], self.iso))

        else:
            matrix = transmissivity_wrapper(self.iza[0], self.n1[0], self.n2[0], self.iso)

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

        self.__ematrix = matrix

    def __store(self):
        # self.I = BRDF
        # self.BSC = BSC
        # self.BRF = BRF
        # self.EMS = Emissivity
        pass

    @property
    def ematrix(self):
        return self.__ematrix

    @property
    def rmatrix(self):
        return self.__rmatrix

    @property
    def tmatrix(self):
        return self.__tmatrix

# def __store_r(self):
#
#     if len(self.iza) > 1:
#         VV = [np.sum(self.__rmatrix[i][0,]) for i in range(self.iza.shape[0])]
#         HH = [np.sum(self.__rmatrix[i][1,]) for i in range(self.iza.shape[0])]
#         VH = [np.sum(self.__rmatrix[i][2,]) for i in range(self.iza.shape[0])]
#         HV = [np.sum(self.__rmatrix[i][3,]) for i in range(self.iza.shape[0])]
#
#         VV = np.asarray(VV).flatten()
#         HH = np.asarray(HH).flatten()
#         VH = np.asarray(VH).flatten()
#         HV = np.asarray(HV).flatten()
#
#     else:
#         VV = np.sum(self.__rmatrix[0,])
#         HH = np.sum(self.__rmatrix[1,])
#         VH = np.sum(self.__rmatrix[2,])
#         HV = np.sum(self.__rmatrix[3,])
#
#     self.I = ReflectanceResult(mat=self.__rmatrix,
#                                VV=VV,
#                                HH=HH,
#                                VH=VH,
#                                HV=HV)


# self.BSC
# self.BRDF
# self.BRF

# class Fresnel(Kernel):
#     def __init__(self, iza, frequency, n1, n2, sigma, normalize=False, nbar=0.0, angle_unit='DEG'):
#
#         self.frequency = frequency
#         self.k0 = 2 * np.pi * frequency / 30
#         self.n1 = n1
#         self.n2 = n2
#         self.sigma = sigma
#
#         raa = np.zeros_like(iza)
#         vza = raa
#
#         super(Fresnel, self).__init__(iza, vza, raa, normalize, nbar, angle_unit)
#
#         # self.rza = self.vza
#         # self.vza = np.zeros_like(self.iza)
#         self.rza = self.snell(self.iza)
#
#         self.reflection()
#         self.transmission()
#         self.reflectivity()
#         self.transmissivity()
#
#     def snell(self, iza):
#         """
#         return angle theta in layer 2 with refractive index n_2, assuming
#         it has angle th_1 in layer with refractive index n_1. Use Snell's law. Note
#         that "angles" may be complex!!
#         """
#         # Important that the arcsin here is scipy.arcsin, not numpy.arcsin! (They
#         # give different results e.g. for arcsin(2).)
#
#         th_2_guess = sp.arcsin(self.n1 * np.sin(iza) / self.n2)
#
#         try:
#             if self.__is_forward_angle(self.n2, th_2_guess):
#
#                 return th_2_guess
#             else:
#                 return pi - th_2_guess
#
#         except AssertionError:
#             return th_2_guess
#
#
#     def __is_forward_angle(self, n, theta):
#         """
#         if a wave is traveling at angle theta from normal in a medium with index n,
#         calculate whether or not this is the forward-traveling wave (i.e., the one
#         going from front to back of the stack, like the incoming or outgoing waves,
#         but unlike the reflected wave). For real n & theta, the criterion is simply
#         -pi/2 < theta < pi/2, but for complex n & theta, it's more complicated.
#         See https://arxiv.org/abs/1603.02720 appendix D. If theta is the forward
#         angle, then (pi-theta) is the backward angle and vice-versa.
#         """
#         assert n.real * n.imag >= 0, ("For materials with gain, it's ambiguous which "
#                                       "beam is incoming vs outgoing. See "
#                                       "https://arxiv.org/abs/1603.02720 Appendix C.\n"
#                                       "n: " + str(n) + "   angle: " + str(theta))
#         ncostheta = n * cos(theta)
#         if abs(ncostheta.imag) > 100 * EPSILON:
#             # Either evanescent decay or lossy medium. Either way, the one that
#             # decays is the forward-moving wave
#             answer = (ncostheta.imag > 0)
#         else:
#             # Forward is the one with positive Poynting vector
#             # Poynting vector is Re[n cos(theta)] for s-polarization or
#             # Re[n cos(theta*)] for p-polarization, but it turns out they're consistent
#             # so I'll just assume s then check both below
#             answer = (ncostheta.real > 0)
#         # convert from numpy boolean to the normal Python boolean
#         answer = bool(answer)
#         # double-check the answer ... can't be too careful!
#         error_string = ("It's not clear which beam is incoming vs outgoing. Weird"
#                         " index maybe?\n"
#                         "n: " + str(n) + "   angle: " + str(theta))
#         if answer is True:
#             assert ncostheta.imag > -100 * EPSILON, error_string
#             assert ncostheta.real > -100 * EPSILON, error_string
#             assert (n * cos(theta.conjugate())).real > -100 * EPSILON, error_string
#         else:
#             assert ncostheta.imag < 100 * EPSILON, error_string
#             assert ncostheta.real < 100 * EPSILON, error_string
#             assert (n * cos(theta.conjugate())).real < 100 * EPSILON, error_string
#         return answer
#
#     def reflection(self):
#         """
#         reflection amplitude (from Fresnel equations)
#         polarization is either "s" or "p" for polarization
#         n_i, n_f are (complex) refractive index for incident and final
#         th_i, th_f are (complex) propegation angle for incident and final
#         (in radians, where 0=normal). "th" stands for "theta".
#         """
#
#         self.rV = ((self.n2 * cos(self.iza) - self.n1 * cos(self.rza)) /
#                    (self.n2 * cos(self.iza) + self.n1 * cos(self.rza)))
#
#         self.rH = ((self.n1 * cos(self.iza) - self.n2 * cos(self.rza)) /
#                    (self.n1 * cos(self.iza) + self.n2 * cos(self.rza)))
#
#     def transmission(self):
#         """
#         transmission amplitude (frem Fresnel equations)
#         polarization is either "s" or "p" for polarization
#         n_i, n_f are (complex) refractive index for incident and final
#         th_i, th_f are (complex) propegation angle for incident and final
#         (in radians, where 0=normal). "th" stands for "theta".
#         """
#
#         n_i = self.n2
#         n_f = self.n1
#
#         self.tV = 2 * n_i * cos(self.iza) / (n_i * cos(self.iza) + n_f * cos(self.rza))
#         self.tH = 2 * n_i * cos(self.iza) / (n_f * cos(self.iza) + n_i * cos(self.rza))
#
#     def reflectivity(self):
#         # matrix = np.zeros((4, 4))
#         #
#         # matrix[0, 0] = abs(self.rV) ** 2
#         # matrix[1, 1] = abs(self.rH) ** 2
#         # matrix[2, 2] = np.real(self.rV * np.conjugate(self.rH))
#         # matrix[2, 3] = -np.imag(self.rV * np.conjugate(self.rH))
#         # matrix[3, 2] = np.imag(self.rV * np.conjugate(self.rH))
#         # matrix[3, 3] = np.real(self.rV * np.conjugate(self.rH))
#
#         h = 4 * self.sigma ** 2 * self.k0 ** 2
#         loss = np.exp(-h * np.cos(self.iza) ** 2)
#
#         # matrix *= loss
#
#         self.ref = ReflectanceResult(V=loss * abs(self.rV) ** 2,
#                                      H=loss * abs(self.rH) ** 2,
#                                      VH=loss * (np.real(self.rV * np.conjugate(self.rH)) - np.imag(
#                                          self.rV * np.conjugate(self.rH))),
#                                      HV=loss * (np.imag(self.rV * np.conjugate(self.rH)) + np.real(
#                                          self.rV * np.conjugate(self.rH))),
#                                      loss=loss, h=h)
#
#     def transmissivity(self):
#         # matrix = np.zeros((4, 4))
#         #
#         # matrix[0, 0] = abs(self.tV) ** 2
#         # matrix[1, 1] = abs(self.tH) ** 2
#         # matrix[2, 2] = np.real(self.tV * np.conjugate(self.tH))
#         # matrix[2, 3] = -np.imag(self.tV * np.conjugate(self.tH))
#         # matrix[3, 2] = np.imag(self.tV * np.conjugate(self.tH))
#         # matrix[3, 3] = np.real(self.tV * np.conjugate(self.tH))
#
#         factor = (self.n2 ** 3 * np.cos(self.rza)).real / (self.n1 ** 3 * np.cos(self.iza)).real
#
#         # matrix *= factor
#
#         self.tra = ReflectanceResult(V=abs(self.tV) ** 2 * factor,
#                                      H=abs(self.tH) ** 2 * factor,
#                                      VH=factor * (np.real(self.tV * np.conjugate(self.tH)) - np.imag(
#                                          self.tV * np.conjugate(self.tH))),
#                                      HV=factor * (np.imag(self.tV * np.conjugate(self.tH)) + np.real(
#                                          self.tV * np.conjugate(self.tH))),
#                                      factor=factor)
#
#     # @staticmethod
#     # def integrate_func(x, frequency, n1, n2, sigma, polarization="V"):
#     #     call = Fresnel(x, frequency, n1, n2, sigma, normalize=False, nbar=0.0, angle_unit='RAD')
#     #     if polarization is 'V':
#     #         return call.ref.V
#     #     elif polarization is 'H':
#     #         return call.ref.H
#     #
#     #     else:
#     #         raise AssertionError("The polarization must be V or H")
