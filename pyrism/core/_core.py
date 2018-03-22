# -*- coding: utf-8 -*-
from __future__ import division

import sys

import numpy as np

from .auxiliary import (rad, deg, sec, align_all, asarrays)

# python 3.6 comparability
if sys.version_info < (3, 0):
    srange = xrange
else:
    srange = range


class Kernel(object):
    """
    The kernel object defines the different models.

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
    align : boolean, optional
         Expand all input values to the same length (default).

    Returns
    -------
    All returns are attributes!
    iza: ndarray
        Sun or incidence zenith angle in [RAD].
    vza : ndarray
        View or scattering zenith angle in [RAD].
    raa : ndarray
        Relative azimuth angle in [RAD].
    izaDeg : ndarray
        Sun or incidence zenith angle in [DEG].
    vzaDeg : ndarray
        View or scattering zenith angle in [DEG].
    raaDeg : ndarray
        Relative azimuth angle in [DEG].
    phi : ndarray
        Relative azimuth angle in a range between 0 and 2pi.

    Note
    ----
    Hot spot direction is vza == iza and raa = 0.0

    """
    def __init__(self, iza, vza, raa, normalize=False, nbar=0.0, angle_unit='DEG', align=True):

        # Initialize values
        self.vza = vza
        self.iza = iza
        self.raa = raa

        self.normalize = normalize
        self.nbar = nbar
        self.angle_unit = angle_unit

        # Assertions
        if self.angle_unit != 'DEG' and self.angle_unit != 'RAD':
            raise AssertionError(
                "angle_unit must be 'DEG' or 'RAD', but angle_unit is: {}".format(str(self.angle_unit)))

        # Initialize angle information
        self.__pre_process(align)
        self.__set_angle()

    def normalization(self, kernel=None, args=None):
        if args is None and kernel is None:
            raise ValueError("kernel or/ and args must be defined.")
        else:
            if args is None:
                kernel = kernel - kernel[-1]
                return kernel

            elif kernel is None:
                return [item[0:-1] for item in args]

            else:
                kernel = kernel - kernel[-1]
                list_args = list(args)
                list_args.append(kernel)
                args = tuple(list_args)
                return [item[0:-1] for item in args]

    def __pre_process(self, align):
        self.iza, self.vza, self.raa = asarrays((self.iza, self.vza, self.raa))

        if align:
            self.iza, self.vza, self.raa = align_all((self.iza, self.vza, self.raa))

        else:
            try:
                if len(self.vza) != len(self.iza) or len(self.vza) != len(self.raa):
                    raise AssertionError("Input dimensions must agree. "
                                         "The actual dimensions are "
                                         "iza: {0}, vza: {1} and raa: {2}".format(str(len(self.iza)),
                                                                                  str(len(self.vza)),
                                                                                  str(len(self.raa))))

            except (AttributeError, TypeError):
                pass

    def __set_angle(self):
        """
        A method to store and organize the input angle data. This also convert
        all angle data in degrees to radians.
        """

        if self.angle_unit is 'DEG':
            self.vzaDeg = self.vza.flatten()
            self.izaDeg = self.iza.flatten()
            self.raaDeg = self.raa.flatten()

            if self.normalize:
                # calculate nadir term by extending array
                self.vzaDeg = np.array(list(self.vzaDeg) + [0.0]).flatten()
                self.izaDeg = np.array(list(self.izaDeg) + [self.nbar]).flatten()
                self.raaDeg = np.array(list(self.raaDeg) + [0.0]).flatten()
                self.B = (sec(np.mean(rad(self.izaDeg[0:-1]))) + sec(np.mean(rad(self.vzaDeg[0:-1]))))
            else:
                self.B = (sec(np.mean(rad(self.izaDeg))) + sec(np.mean(rad(self.vzaDeg))))

            self.vza = rad(self.vzaDeg)
            self.iza = rad(self.izaDeg)
            self.raa = rad(self.raaDeg)

            # Check if there are negative angle values
            w = np.where(self.vza < 0)[0]
            self.vza[w] = -self.vza[w]
            self.raa[w] = self.raa[w] + np.pi
            w = np.where(self.iza < 0)[0]
            self.iza[w] = -self.iza[w]
            self.raa[w] = self.raa[w] + np.pi

            # Turn the raa values in to a range between 0 and 2*pi
            self.phi = np.abs((self.raa % (2. * np.pi)))

        if self.angle_unit is 'RAD':
            self.vza = self.vza.flatten()
            self.iza = self.iza.flatten()
            self.raa = self.raa.flatten()

            if self.normalize:
                # calculate nadir term by extending array
                self.vza = np.array(list(self.vza) + [0.0]).flatten()
                self.iza = np.array(list(self.iza) + [self.nbar]).flatten()
                self.raa = np.array(list(self.raa) + [0.0]).flatten()
                self.B = (sec(np.mean(self.iza[0:-1])) + sec(np.mean(self.vza[0:-1])))

            else:
                self.B = (sec(np.mean(self.iza)) + sec(np.mean(self.vza)))

            # Check if there are negative angle values
            w = np.where(self.vza < 0)[0]
            self.vza[w] = -self.vza[w]
            self.raa[w] = self.raa[w] + np.pi
            w = np.where(self.iza < 0)[0]
            self.iza[w] = -self.iza[w]
            self.raa[w] = self.raa[w] + np.pi

            self.vzaDeg = deg(self.vza)
            self.izaDeg = deg(self.iza)
            self.raaDeg = deg(self.raa)

            # Turn the raa values in to a range between 0 and 2 pi
            self.phi = np.abs((self.raa % (2. * np.pi)))


class Scattering(object):
    """

    Calculate the extinction coefficients in terms of Rayleigh or Mie
    scattering from :cite:`Ulaby.2015` and :cite:`Ulaby.2015b`.

    Parameters
    ----------
    frequency : int or float
        Frequency (GHz)

    particle_size : int, float or array
        Particle size a (cm).

    diel_constant_p : complex
        Dielectric constant of the medium.

    diel_constant_b : complex
        Dielectric constant of the background.

    """
    def __init__(self, frequency, particle_size, diel_constant_p, diel_constant_b):
        frequency, particle_size, diel_constant_p, diel_constant_b = asarrays(
            (frequency, particle_size, diel_constant_p, diel_constant_b))


        self.freq = frequency
        self.a = particle_size
        self.er_p = diel_constant_p
        self.er_b = diel_constant_b
        self.__pre_process()

    def __pre_process(self):
        self.er_b_real = self.er_b.real
        self.np = np.sqrt(self.er_p)  # index of refraction of spherical particle
        self.nb = np.sqrt(self.er_b)  # index of refraction of background medium
        self.n = self.np / self.nb
        self.chi = (20 / 3) * np.pi * self.a * self.freq * np.sqrt(self.er_b_real)
