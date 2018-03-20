# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np

from .auxiliary import (rad, deg, sec, align_all, asarrays)

# python 3.6 comparability
try:
    xrange
except NameError:
    xrange = range


class Kernel(object):

    def __init__(self, iza, vza, raa, normalize=False, nbar=0.0, angle_unit='DEG', align=True):
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

        Attributes
        ----------
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

        Notes
        -----
        Hot spot direction is vza == iza and raa = 0.0

        """

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

    @staticmethod
    def check_data(angle, values):

        if len(angle) == 1 and len(values) > 1:
            x = True

        elif len(angle) > 1 and len(values) == 1:
            x = True

        elif len(angle) == 1 and len(values) == 1:
            x = True

        elif len(angle) > 1 and len(values) > 1:
            if len(angle) == len(values):
                x = True

            else:
                x = False
        else:
            x = False

        if not x:
            raise ValueError("If the length of iza, vza or raa is greater than 1 and the length of values "
                             "(e.g. values.d) is greater than 1 the input dimensions of iza, "
                             "vza, raa and values must agree."
                             "The actual dimensions are at angle:{0} and values: {1}".format(str(len(angle)),
                                                                                             str(len(values))))

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

    @staticmethod
    def get_distance_function(iza, vza, phi):
        """
        A method to set the distance component of geometric kernels.

        Parameters:
        ----------
        iza : int, float or ndarray
            Sun or incidence zenith angle.

        vza : int, float or ndarray
            View or scattering zenith angle.

        phi : int, float or ndarray
            Relative azimuth angle in a range between 0 and 2pi.

        Returns:
        --------
        inner_distance: int, float or ndarray
            Distance component before taking the square root.

        distance: int, float or ndarray
            Square root of inner_distance.
        """

        inner_distance = np.tan(iza) * np.tan(iza) + np.tan(vza) * np.tan(vza) - 2. * np.tan(iza) * np.tan(
            vza) * np.cos(phi)

        try:
            w = np.where(inner_distance < 0)[0]
            inner_distance[w] = 0.0

        except TypeError:
            if inner_distance < 0:
                inner_distance = 0

        distance = np.sqrt(inner_distance)

        return inner_distance, distance

    @staticmethod
    def get_proj_angle(br_ratio, xza):
        """
        Method to do B/R transformation for ellipse shape.

        Parameters:
        -----------
        br_ratio : int, float or ndarray
            Relations between the height of the spheroid and their radius (b/r).

        xza : int, float or ndarray
            Any zenith angle in [RAD].

        Returns:
        --------
        angp : int, float or ndarray
            Projected and transformed angle
        """

        t = br_ratio * np.tan(xza)

        try:
            w = np.where(t < 0.)[0]
            t[w] = 0.0

        except TypeError:
            if t < 0:
                t = 0
            else:
                pass

        angp = np.arctan(t)
        return angp

    @classmethod
    def get_overlap(cls, hb_ratio, iza, vza, phi):
        """
        Method to do HB ratio transformation

        Parameters:
        -----------
        hb_ratio : int, float or ndarray
            Relations between the height of the stick under the spheroid (h/b)

        iza : int, float or ndarray
            Sun or incidence zenith angle.

        vza : int, float or ndarray
            View or scattering zenith angle.

        phi : int, float or ndarray
            Relative azimuth angle in a range between 0 and 2pi.

        Returns:
        --------
        overlap : int, float or ndarray
        """
        B = sec(iza) + sec(vza)

        _, distance = cls.get_distance_function(iza, vza, phi)

        cost = hb_ratio * np.sqrt(distance ** 2 + np.tan(vza) ** 2 * np.tan(iza) ** 2 * np.sin(phi) ** 2) / B
        try:
            w = np.where(cost < -1)[0]
            cost[w] = -1.0
            w = np.where(cost > 1.0)[0]
            cost[w] = 1.0
            tvar = np.arccos(cost)
            sint = np.sin(tvar)
            overlap = (1 / np.pi) * (tvar - sint * cost) * B
            w = np.where(overlap < 0)[0]
            overlap[w] = 0.0

        except TypeError:
            if cost < -1:
                cost = -1

            elif cost > 1:
                cost = 1

            else:
                pass

            tvar = np.arccos(cost)
            sint = np.sin(tvar)
            overlap = (1 / np.pi) * (tvar - sint * cost) * B

            if overlap < 0:
                overlap = 0
            else:
                pass

        return overlap


class Scattering(object):
    """

    Calculate the extinction coefficients in terms of Rayleigh or Mie
    scattering.

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

    References
    ----------
    .. [1] ULABY, F. T., LONG, D. G., BLACKWELL, W. J., ELACHI, C., FUNG, A. K.,
            RUF, C., SARABANDI, K., ZEBKER, H. A., & VAN ZYL, J. (2014).
            Microwave radar and radiometric remote sensing. Ann Arbor,
            University of Michigan Press.

    .. [2] http://mrs.eecs.umich.edu/microwave_remote_sensing_computer_codes.html

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
