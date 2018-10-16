from __future__ import division

import sys
import warnings

warnings.simplefilter('default')

import numpy as np
from pyrism.core.tma import (calc_nmax_wrapper, get_oriented_SZ, sca_xsect_wrapper, asym_wrapper, ext_xsect,
                             sca_intensity_wrapper, dblquad_SZ_wrapper, dblquad_oriented_SZ_wrapper)
from radarpy import Angles, align_all

from .orientation import Orientation
from .tm_auxiliary import param

# python 3.6 comparability
if sys.version_info < (3, 0):
    srange = xrange
else:
    srange = range


class TMatrixSingle(Angles, object):
    def __init__(self, iza, vza, iaa, vaa, frequency, radius, eps, alpha=0.0, beta=0.0,
                 radius_type='REV', shape='SPH', orientation='S', axis_ratio=1.0, orientation_pdf=None, n_alpha=5,
                 n_beta=10, angle_unit='DEG', normalize=False, nbar=0.0):

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
        normalize : boolean, optional
            Set to 'True' to make kernels 0 at nadir view illumination. Since all implemented kernels are normalized
            the default value is False.
        nbar : float, optional
            The sun or incidence zenith angle at which the isotropic term is set
            to if normalize is True. The default value is 0.0.

        Returns
        -------
        TMatrixSingle.S : array_like
            Complex Scattering Matrix.
        TMatrixSingle.Z : array_like
            Phase Matrix.
        TMatrixSingle.SZ : tuple
             Complex Scattering Matrix and Phase Matrix.
        TMatrixSingle.ksi : tuple
            Scattering intensity for VV and HH polarization.
        TMatrixSingle.ksx : tuple
            Scattering Cross Section for VV and HH polarization.
        TMatrixSingle.kex : tuple
            Extinction Cross Section for VV and HH polarization.
        TMatrixSingle.asx : tuple
            Asymetry Factor for VV and HH polarization.

        See Also
        --------
        radarpy.Angles

        """
        # ---- Define angles and align data ----
        iza, vza, iaa, vaa, frequency, radius, axis_ratio, alpha, beta = align_all(
            (iza, vza, iaa, vaa, frequency, radius, axis_ratio, alpha, beta))

        _, eps = align_all((iza, eps))

        Angles.__init__(self, iza=iza, vza=vza, raa=None, iaa=iaa, vaa=vaa, alpha=alpha, beta=beta,
                        normalize=normalize, angle_unit=angle_unit, nbar=nbar)

        if normalize:
            _, frequency, radius, axis_ratio, alpha, beta = align_all(
                (self.iza, frequency, radius, axis_ratio, alpha, beta))

        self.normalize = normalize
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

        nmax = self.__calc_nmax()
        self.nmax = nmax.astype(np.float)
        # self.__S, self.__Z = self.call_SZ()

    # ---- Property calls ----
    @property
    def S(self):
        """
        Scattering matrix.

        Returns
        -------
        S : list or array_like
        """
        try:
            if len(self.__S) == 1:
                return self.__S[0]

            else:
                if self.normalize:
                    if len(self.__S[0:-1]) == 1:
                        return self.__S[0:-1][0]

                    else:
                        return self.__S[0:-1][0]

                else:
                    return self.__S

        except AttributeError:

            self.__S, self.__Z = self.__call_SZ()

            if len(self.__S) == 1:
                return self.__S[0]
            else:
                if self.normalize:
                    if len(self.__S[0:-1]) == 1:
                        return self.__S[0:-1][0]

                    else:
                        return self.__S[0:-1][0]

                else:
                    return self.__S

    @property
    def norm(self):
        """
        Normalization matrix. The values for iza = nbar, vza = 0.

        Returns
        -------
        Norm : list or array_like
        """
        if self.normalize:
            try:
                return self.__Z[-1]

            except AttributeError:

                self.__S, self.__Z = self.__call_SZ()

                return self.__Z[-1]

    @property
    def dblquad(self):
        """
        Half space integration of the phase matrices.

        Returns
        -------
        dbl : list or array_like
        """
        try:
            if len(self.__dblquadZ) == 1:
                return self.__dblquadZ[0]
            else:
                if self.normalize:
                    if len(self.__dblquadZ[0:-1]) == 1:
                        return self.__dblquadZ[0:-1][0] - self.__dblquadZ[-1]
                    else:
                        return self.__dblquadZ[0:-1] - self.__dblquadZ[-1]

                else:
                    return self.__dblquadZ

        except AttributeError:

            self.__dblquadZ = self.__dblquad()

            if len(self.__dblquadZ) == 1:
                return self.__dblquadZ[0]
            else:
                if self.normalize:
                    if len(self.__dblquadZ[0:-1]) == 1:
                        return self.__dblquadZ[0:-1][0] - self.__dblquadZ[-1]
                    else:
                        return self.__dblquadZ[0:-1] - self.__dblquadZ[-1]

                else:
                    return self.__dblquadZ

    @property
    def Z(self):
        """
        Phase matrix.

        Returns
        -------
        Z : list or array_like
        """
        try:
            if len(self.__Z) == 1:
                return self.__Z[0]
            else:
                if self.normalize:
                    if len(self.__Z[0:-1]) == 1:
                        return self.__Z[0:-1][0] - self.__Z[-1]
                    else:
                        return self.__Z[0:-1] - self.__Z[-1]

                else:
                    return self.__Z

        except AttributeError:

            self.__S, self.__Z = self.__call_SZ()

            if len(self.__Z) == 1:
                return self.__Z[0]
            else:
                if self.normalize:
                    if len(self.__Z[0:-1]) == 1:
                        return self.__Z[0:-1][0] - self.__Z[-1]
                    else:
                        return self.__Z[0:-1] - self.__Z[-1]

                else:
                    return self.__Z

    @property
    def SZ(self):
        """
        Scattering and phase matrices.

        Returns
        -------
        S, Z : list or array_like
        """
        return self.S, self.Z

    # ---- User callable methods ----
    def ifunc_SZ(self, izaDeg, iaaDeg, pol):
        """
        Function to integrate the phase matrix which is compatible with scipy.integrate.dblquad.

        Parameters
        ----------
        izaDeg : float
            x value of integration (0, 180) in [DEG].
        iaaDeg : float
            y value of integration (0, 360) in [DEG]
        pol : bool
            Which polarization should be integrated?
                * 0 : VV
                * 1 : HH

        Examples
        --------
        scipy.integrate.dblquad(TMatrix.ifunc_SZ, 0, 360.0, lambda x: 0.0, lambda x: 180.0, args=(0, ))

        Returns
        -------
        Z : float

        """
        if len(self.izaDeg) > 1:
            warnings.warn(
                "The length of the input parameter are greater than 1. The first element will be used for the integration.")
        Z = dblquad_oriented_SZ_wrapper(izaDeg, iaaDeg, self.nmax[0], self.wavelength[0], self.vzaDeg[0],
                                        self.vaaDeg[0], self.alphaDeg[0], self.betaDeg[0], self.n_alpha,
                                        self.n_beta, self.or_pdf, self.orient, pol)

        return Z

    def ksi(self):
        """
        Scattering intensity (phase function) for the current setup.

        Returns
        -------
        VV, HH : list or array_like

        """
        VV_list = list()
        HH_list = list()
        for i in srange(len(self.iza)):
            VV = sca_intensity_wrapper(self.__Z[i], 1)
            HH = sca_intensity_wrapper(self.__Z[i], 2)

            VV_list.append(VV)
            HH_list.append(HH)

        return np.asarray(VV_list).flatten(), np.asarray(HH_list).flatten()

    def ksx(self):
        """
        Scattering cross section for the current setup, with polarization.

        Returns
        -------
        VV, HH : list or array_like

        """

        xsectVV_list = list()
        xsectHH_list = list()
        for i in srange(len(self.iza)):
            xsectVV, xsectHH = sca_xsect_wrapper(self.nmax[i],
                                                 self.wavelength[i], self.izaDeg[i],
                                                 self.iaaDeg[i], self.alphaDeg[i], self.betaDeg[i],
                                                 self.n_alpha, self.n_beta,
                                                 self.or_pdf, self.orient)

            xsectVV_list.append(xsectVV)
            xsectHH_list.append(xsectHH)

        return np.asarray(xsectVV_list).flatten(), np.asarray(xsectHH_list).flatten()

    def kex(self):
        """
        Extinction cross section for the current setup, with polarization.

        Returns
        -------
        VV, HH : list or array_like

        """
        VV_list = list()
        HH_list = list()
        for i in srange(len(self.iza)):
            VV, HH = ext_xsect(self.nmax[i], self.wavelength[i], self.izaDeg[i], self.vzaDeg[i],
                               self.iaaDeg[i], self.vaaDeg[i], self.alphaDeg[i],
                               self.betaDeg[i], self.n_alpha, self.n_beta, self.or_pdf,
                               self.orient)

            VV_list.append(VV)
            HH_list.append(HH)

        return np.asarray(VV_list).flatten(), np.asarray(HH_list).flatten()

    def asx(self):
        """
        Asymetry factor cross section for the current setup, with polarization.

        Returns
        -------
        VV, HH : list or array_like

        """
        xsectVV_list = list()
        xsectHH_list = list()
        for i in srange(len(self.iza)):
            xsectVV, xsectHH = asym_wrapper(self.nmax[i],
                                            self.wavelength[i], self.izaDeg[i],
                                            self.iaaDeg[i], self.alphaDeg[i], self.betaDeg[i],
                                            self.n_alpha, self.n_beta,
                                            self.or_pdf, self.orient)

            xsectVV_list.append(xsectVV)
            xsectHH_list.append(xsectHH)

        return np.asarray(xsectVV_list).flatten(), np.asarray(xsectHH_list).flatten()

    # ---- Auxiliary functions and private methods ----
    def __calc_nmax(self):
        """Initialize the T-matrix and calculate nmax parameter.
        """

        nmax = list()
        for i in srange(len(self.izaDeg)):
            temp = calc_nmax_wrapper(self.radius[i], self.radius_type, self.wavelength[i], self.eps[i],
                                     self.axis_ratio[i], self.shape)

            nmax.append(temp)

        return np.asarray(nmax).flatten()

    def __call_SZ(self):
        """Get the S and Z matrices for a orientated SZ.
        """
        S_list = list()
        Z_list = list()

        for i in srange(len(self.izaDeg)):
            S, Z = get_oriented_SZ(self.nmax[i], self.wavelength[i], self.izaDeg[i], self.vzaDeg[i],
                                   self.iaaDeg[i], self.vaaDeg[i], self.alphaDeg[i],
                                   self.betaDeg[i], self.n_alpha, self.n_beta, self.or_pdf,
                                   self.orient)

            S_list.append(S)
            Z_list.append(Z)

        return S_list, Z_list

    def __dblquad(self):
        """Call the integrate function for half space integration.

        Returns
        -------
        Z : array_like
            Integrated Phase matrix.
        """
        Z_list = list()

        for i in srange(len(self.izaDeg)):
            Z = np.zeros((2, 2))
            Z[0, 0] = dblquad_SZ_wrapper(self.nmax[i], self.wavelength[i], self.vzaDeg[i],
                                         self.vaaDeg[i], self.alphaDeg[i],
                                         self.betaDeg[i], self.n_alpha, self.n_beta, self.or_pdf,
                                         self.orient, 0)

            Z[1, 1] = dblquad_SZ_wrapper(self.nmax[i], self.wavelength[i], self.vzaDeg[i],
                                         self.vaaDeg[i], self.alphaDeg[i],
                                         self.betaDeg[i], self.n_alpha, self.n_beta, self.or_pdf,
                                         self.orient, 1)

            # S_list.append(S)
            Z_list.append(Z)

        self.__dblquadZ = Z_list

        return self.__dblquadZ

    def __get_pdf(self, pdf):
        """
        Auxiliary function to determine the PDF function.

        Parameters
        ----------
        pdf : {'gauss', 'uniform'}
            Particle orientation Probability Density Function (PDF) for orientational averaging:
                * 'gauss': Use a Gaussian PDF (default).
                * 'uniform': Use a uniform PDR.

        Returns
        -------
        function : callable

        """
        if callable(pdf):
            return pdf
        elif pdf is None:
            return Orientation.gaussian()
        else:
            raise AssertionError(
                "The Particle size distribution (psd) must be callable or 'None' to get the default gaussian psd.")
