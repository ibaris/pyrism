from __future__ import division

import numpy as np
from pyrism.core.tma import (calc_nmax_wrapper, get_oriented_SZ, sca_xsect_wrapper, asym_wrapper, ext_xsect,
                             sca_intensity_wrapper)

from radarpy import Angles, asarrays, align_all

from .orientation import Orientation


class TMatrixSingle(Angles, object):
    def __init__(self, iza, vza, iaa, vaa, frequency, radius, eps, alpha=0.0, beta=0.0,
                 radius_type='REV', shape='SPH', orientation='S', axis_ratio=1.0, orientation_pdf=None, n_alpha=5,
                 n_beta=10,
                 angle_unit='DEG'):

        """T-Matrix scattering from single nonspherical particles.

        Class for simulating scattering from nonspherical particles with the
        T-Matrix method. Uses a wrapper to the Fortran code by M. Mishchenko.

        Parameters
        ----------
        iza, vza, iaa, vaa : int, float or array_like
            Incidence (iza) and scattering (vza) zenith angle and incidence and viewing
            azimuth angle (ira, vra) in [DEG] or [RAD] (see parameter angle_unit).
        frequency : int float or array_like
            The frequency of incident EM Wave in [GHz].
        radius : int float or array_like
            Equivalent particle radius in [cm].
        radius_type : {'EV', 'M', 'REA'}
            Specification of particle radius:
                * 'REV': radius is the equivalent volume radius (default).
                * 'M': radius is the maximum radius.
                * 'REA': radius is the equivalent area radius.
        eps : complex
            The complex refractive index.
        axis_ratio : int or float
            The horizontal-to-rotational axis ratio.
        shape : {'SPH', 'CYL'}
            Shape of the particle:
                * 'SPH' : spheroid,
                * 'CYL' : cylinders.
        alpha, beta: int, float or array_like
            The Euler angles of the particle orientation in [DEG] or [RAD] (see parameter angle_unit).
        orientation : {'S', 'AA', 'AF'}
            The function to use to compute the orientational scattering properties:
                * 'S': Single (default).
                * 'AA': Averaged Adaptive
                * 'AF': Averaged Fixed.
        orientation_pdf: {'gauss', 'uniform'}
            Particle orientation Probability Density Function (PDF) for orientational averaging:
                * 'gauss': Use a Gaussian PDF (default).
                * 'uniform': Use a uniform PDR.
        n_alpha : int
            Number of integration points in the alpha Euler angle. Default is 5.
        n_beta : int
            Umber of integration points in the beta Euler angle. Default is 10.
        angle_unit : {'DEG', 'RAD'}, optional
            * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
            * 'RAD': All input angles (iza, vza, raa) are in [RAD].

        Returns
        -------
        TMatrixSingle.S : array_like
            Complex Scattering Matrix.
        TMatrixSingle.Z : array_like
            Phase Matrix.
        TMatrixSingle.SZ : tuple
             Complex Scattering Matrix and Phase Matrix.
        TMatrixSingle.get

        """
        param = {'REV': 1.0,
                 'REA': 0.0,
                 'M': 2.0,
                 'SPH': -1,
                 'CYL': -2}

        iza, vza, iaa, vaa, frequency, radius, axis_ratio, alpha, beta = asarrays(
            (iza, vza, iaa, vaa, frequency, radius, axis_ratio, alpha, beta))

        eps = np.asarray(eps).flatten()

        iza, vza, iaa, vaa, frequency, radius, axis_ratio, alpha, beta = align_all(
            (iza, vza, iaa, vaa, frequency, radius, axis_ratio, alpha, beta))

        Angles.__init__(self, iza=iza, vza=vza, raa=None, iaa=iaa, vaa=vaa, alpha=alpha, beta=beta,
                        normalize=False, angle_unit=angle_unit)

        # super(TMatrixSingle, self).__init__(iza=iza, vza=vza, raa=None, iaa=iaa, vaa=vaa, alpha=alpha, beta=beta,
        #                                     normalize=False,
        #                                     angle_unit=angle_unit)

        self.radius = radius
        self.radius_type = param[radius_type]
        self.frequency = frequency

        self.wavelength = 29.9792458 / frequency
        self.axis_ratio = axis_ratio
        self.shape = param[shape]
        self.ddelt = 1e-3
        self.ndgs = 2
        self.alpha = alpha
        self.beta = beta
        self.orient = orientation

        self.or_pdf = self.__get_pdf(orientation_pdf)
        self.n_alpha = int(n_alpha)
        self.n_beta = int(n_beta)

        self.eps, _ = align_all((eps, self.iza))

        self.nmax = self.__calc_nmax()
        # self.__S, self.__Z = self.call_SZ()

    @property
    def S(self):
        try:
            if len(self.__S) == 1:
                return self.__S[0]
            else:
                return self.__S

        except AttributeError:

            self.__S, self.__Z = self.__call_SZ()

            if len(self.__S) == 1:
                return self.__S[0]
            else:
                return self.__S

    @property
    def Z(self):
        try:
            if len(self.__Z) == 1:
                return self.__Z[0]
            else:
                return self.__Z

        except AttributeError:

            self.__S, self.__Z = self.__call_SZ()

            if len(self.__Z) == 1:
                return self.__Z[0]
            else:
                return self.__Z

    @property
    def SZ(self):
        try:
            if len(self.__S) == 1:
                return self.__S[0], self.__Z[0]
            else:
                return self.__S, self.__Z

        except AttributeError:

            self.__S, self.__Z = self.__call_SZ()

            if len(self.__S) == 1:
                return self.__S[0], self.__Z[0]
            else:
                return self.__S, self.__Z

    def __calc_nmax(self):
        """Initialize the T-matrix.
        """

        nmax = list()
        for i in range(len(self.izaDeg)):
            temp = calc_nmax_wrapper(self.radius[i], self.radius_type, self.wavelength[i], self.eps[i],
                                     self.axis_ratio[i], self.shape)

            nmax.append(temp)

        return np.asarray(nmax).flatten()

    def __call_SZ(self):
        """Get the S and Z matrices for a single orientation.
        """
        S_list = list()
        Z_list = list()

        for i in range(len(self.izaDeg)):
            S, Z = get_oriented_SZ(self.nmax[i], self.wavelength[i], self.izaDeg[i], self.vzaDeg[i],
                                   self.iaaDeg[i], self.vaaDeg[i], self.alphaDeg[i],
                                   self.betaDeg[i], self.n_alpha, self.n_beta, self.or_pdf,
                                   self.orient)

            S_list.append(S)
            Z_list.append(Z)

        return S_list, Z_list

    def ksi(self):
        """Scattering intensity (phase function) for the current setup.

        Args:
            scatterer: a Scatterer instance.
            h_pol: If True (default), use horizontal polarization.
            If False, use vertical polarization.

        Returns:
            The differential scattering cross section.
        """
        VV_list = list()
        HH_list = list()
        for i in range(len(self.iza)):
            VV = sca_intensity_wrapper(self.__Z[i], 1)
            HH = sca_intensity_wrapper(self.__Z[i], 2)

            VV_list.append(VV)
            HH_list.append(HH)

        return np.asarray(VV_list).flatten(), np.asarray(HH_list).flatten()

    def ksx(self):
        """Scattering cross section for the current setup, with polarization.

        Args:
            scatterer: a Scatterer instance.
            h_pol: If True (default), use horizontal polarization.
            If False, use vertical polarization.

        Returns:
            The scattering cross section.
        """

        xsectVV_list = list()
        xsectHH_list = list()
        for i in range(len(self.iza)):
            xsectVV, xsectHH = sca_xsect_wrapper(self.nmax[i],
                                                 self.wavelength[i], self.izaDeg[i],
                                                 self.iaaDeg[i], self.alphaDeg[i], self.betaDeg[i],
                                                 self.n_alpha, self.n_beta,
                                                 self.or_pdf, self.orient)

            xsectVV_list.append(xsectVV)
            xsectHH_list.append(xsectHH)

        return np.asarray(xsectVV_list).flatten(), np.asarray(xsectHH_list).flatten()

    def kex(self):
        VV_list = list()
        HH_list = list()
        for i in range(len(self.iza)):
            VV, HH = ext_xsect(self.nmax[i], self.wavelength[i], self.izaDeg[i], self.vzaDeg[i],
                               self.iaaDeg[i], self.vaaDeg[i], self.alphaDeg[i],
                               self.betaDeg[i], self.n_alpha, self.n_beta, self.or_pdf,
                               self.orient)

            VV_list.append(VV)
            HH_list.append(HH)

        return np.asarray(VV_list).flatten(), np.asarray(HH_list).flatten()

    def __get_pdf(self, pdf):
        if callable(pdf):
            return pdf
        elif pdf is None:
            return Orientation.gaussian()
        else:
            raise AssertionError(
                "The Particle size distribution (psd) must be callable or 'None' to get the default gaussian psd.")

    def asx(self):
        xsectVV_list = list()
        xsectHH_list = list()
        for i in range(len(self.iza)):
            xsectVV, xsectHH = asym_wrapper(self.nmax[i],
                                            self.wavelength[i], self.izaDeg[i],
                                            self.iaaDeg[i], self.alphaDeg[i], self.betaDeg[i],
                                            self.n_alpha, self.n_beta,
                                            self.or_pdf, self.orient)

            xsectVV_list.append(xsectVV)
            xsectHH_list.append(xsectHH)

        return np.asarray(xsectVV_list).flatten(), np.asarray(xsectHH_list).flatten()
