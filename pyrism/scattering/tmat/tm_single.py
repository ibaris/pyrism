from __future__ import division

import sys
import warnings

import numpy as np
from pyrism.core.tma import (NMAX_VEC_WRAPPER, SZ_S_VEC_WRAPPER, SZ_AF_VEC_WRAPPER, DBLQUAD_Z_S_WRAPPER,
                             XSEC_QS_S_WRAPPER, XSEC_ASY_S_WRAPPER, XSEC_ASY_AF_WRAPPER, XSEC_QE_WRAPPER,
                             XSEC_QSI_WRAPPER,
                             equal_volume_from_maximum_wrapper)
from radarpy import Angles, align_all, wavelength

from .orientation import Orientation
from .tm_auxiliary import param

# python 3.6 comparability
if sys.version_info < (3, 0):
    srange = xrange
else:
    srange = range

warnings.simplefilter('default')

PI = 3.14159265359
RAD_TO_DEG = 180.0 / PI
DEG_TO_RAD = PI / 180.0


class TMatrixSingle(Angles, object):
    def __init__(self, iza, vza, iaa, vaa, frequency, radius, eps, alpha=0.0, beta=0.0,
                 radius_type='REV', shape='SPH', orientation='S', axis_ratio=1.0, orientation_pdf=None, n_alpha=5,
                 n_beta=10, normalize=False, nbar=0.0, angle_unit='DEG', frequency_unit='GHz', radius_unit='m',
                 verbose=False):

        """T-Matrix scattering from single nonspherical particles.

        Class for simulating scattering from nonspherical particles with the
        T-Matrix method. Uses a wrapper to the Fortran code by M. Mishchenko.

        Parameters
        ----------
        iza, vza, iaa, vaa : int, float or array_like
            Incidence (iza) and scattering (vza) zenith angle and incidence and viewing
            azimuth angle (ira, vra) in [DEG] or [RAD] (see parameter angle_unit).
        frequency : int float or array_like
            The frequency of incident EM Wave in {'Hz', 'MHz', 'GHz', 'THz'} (see parameter frequency_unit).
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
        frequency_unit : {'Hz', 'MHz', 'GHz', 'THz'}
            Unit of entered frequency. Default is 'GHz'.
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

        # --------------------------------------------------------------------------------------------------------------
        # Define angles and align data
        # --------------------------------------------------------------------------------------------------------------
        iza, vza, iaa, vaa, frequency, radius, axis_ratio, alpha, beta, eps_real, eps_imag = align_all(
            (iza, vza, iaa, vaa, frequency, radius, axis_ratio, alpha, beta, eps.real, eps.imag), dtype=np.double)

        Angles.__init__(self, iza=iza, vza=vza, raa=None, iaa=iaa, vaa=vaa, alpha=alpha, beta=beta,
                        normalize=normalize, angle_unit=angle_unit, nbar=nbar)

        if normalize:
            _, frequency, radius, axis_ratio, alpha, beta, eps_real, eps_imag = align_all(
                (self.iza, frequency, radius, axis_ratio, alpha, beta, eps.real, eps.imag))

        # --------------------------------------------------------------------------------------------------------------
        # Self Definitions
        # --------------------------------------------------------------------------------------------------------------
        self.verbose = verbose
        self.normalize = normalize
        self.radius = radius
        self.radius_type = param[radius_type]
        self.frequency = frequency
        self.radius_unit = radius_unit
        self.frequency_unit = frequency_unit

        self.wavelength = wavelength(self.frequency, unit=self.frequency_unit, output=self.radius_unit)
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

        self.epsi = eps_imag
        self.epsr = eps_real

        # --------------------------------------------------------------------------------------------------------------
        # Calculations
        # --------------------------------------------------------------------------------------------------------------
        self.nmax = self.__NMAX()
        self.xmax = self.nmax.shape[0]
        # self.nmax = nmax.astype(np.float)

    # ------------------------------------------------------------------------------------------------------------------
    # Property Calls
    # ------------------------------------------------------------------------------------------------------------------
    # Scattering and Phase Matrices ------------------------------------------------------------------------------------
    @property
    def S(self):
        """
        Scattering matrix.

        Returns
        -------
        S : array_like
        """
        try:
            return self.__property_return(self.__S, normalize=False)

        except AttributeError:
            self.__S, self.__Z = self.compute_SZ()

            return self.__property_return(self.__S, normalize=False)

    @property
    def Z(self):
        """
        Phase matrix.

        Returns
        -------
        Z : array_like
        """
        try:
            return self.__property_return(self.__Z)

        except AttributeError:
            self.__S, self.__Z = self.compute_SZ()

            return self.__property_return(self.__Z)

    @property
    def SZ(self):
        """
        Scattering and phase matrix.

        Returns
        -------
        S, Z : tuple
        """
        try:
            return self.__property_return(self.__S, normalize=False), self.__property_return(self.__Z)

        except AttributeError:
            self.__S, self.__Z = self.compute_SZ()

            return self.__property_return(self.__S, normalize=False), self.__property_return(self.__Z)

    @property
    def Znorm(self):
        """
        Normalization matrices of the phase matrix.
        Normalization adds an extra column with S and Z values for iza = nbar and vza = 0.

        Returns
        -------
        Z : list or array_like
        """
        if self.normalize:

            try:
                Z = self.__Z
            except AttributeError:
                _, Z = self.compute_SZ()

            try:
                return self.__Znorm

            except AttributeError:
                self.__Znorm = np.zeros((4, 4))

                for i in range(4):
                    for j in range(4):
                        self.__Znorm[i, j] = Z[-1, i, j]

                return self.__Znorm
        return None

    @property
    def Snorm(self):
        """
        Normalization matrices of the scattering matrix.
        Normalization adds an extra column with S and Z values for iza = nbar and vza = 0.

        Returns
        -------
        S, Z : list or array_like
        """
        if self.normalize:

            try:
                S = self.__S
            except AttributeError:
                S, _ = self.compute_SZ()

            try:
                return self.__Snorm

            except AttributeError:
                self.__Snorm = np.zeros((2, 2), dtype=np.complex)

                for i in range(2):
                    for j in range(2):
                        self.__Snorm[i, j] = S[-1, i, j]

                return self.__Snorm
        return None

    # Integration of S and Z -------------------------------------------------------------------------------------------
    @property
    def dblquad(self):
        """
        Half space integration of the phase matrix in incidence direction.

        Returns
        -------
        dbl : list or array_like
        """
        try:
            return self.__property_return(self.__dblZi)

        except AttributeError:
            self.__dblZi = self.__dblquad()

            return self.__property_return(self.__dblZi)

    # Cross Section ----------------------------------------------------------------------------------------------------
    @property
    def QS(self):
        """
        Scattering cross section for the current setup, with polarization.

        Returns
        -------
        QS : array_like
        """

        try:
            return self.__property_return(self.__XS, normalize=False)

        except AttributeError:
            self.__XS = self.__QS()

            return self.__property_return(self.__XS, normalize=False)

    @property
    def QAS(self):
        """
        Asymmetry cross section for the current setup, with polarization.

        Returns
        -------
        QAS : array_like
        """
        try:
            XS = self.__XS

        except AttributeError:
            XS = self.__QS()

        try:
            XAS = self.__XAS

        except AttributeError:
            self.__XAS = self.__QAS()
            XAS = self.__XAS

        return self.__property_return(XAS.base / XS.base, normalize=False)

    @property
    def QE(self):
        """
        Extinction cross section for the current setup, with polarization.

        Returns
        -------
        QE : array_like
        """

        try:
            return self.__property_return(self.__XE, normalize=False)

        except AttributeError:
            self.__XE = self.__QE()

            return self.__property_return(self.__XE, normalize=False)

    @property
    def I(self):
        """
        Extinction cross section for the current setup, with polarization.

        Returns
        -------
        QE : array_like
        """

        try:
            return self.__property_return(self.__XI, normalize=False)

        except AttributeError:
            self.__XI = self.__I()

            return self.__property_return(self.__XI, normalize=False)

    # -----------------------------------------------------------------------------------------------------------------
    # User callable methods
    # -----------------------------------------------------------------------------------------------------------------
    def compute_SZ(self, izaDeg=None, vzaDeg=None, iaaDeg=None, vaaDeg=None, alphaDeg=None, betaDeg=None):
        """T-Matrix scattering from single nonspherical particles.

        Class for simulating scattering from nonspherical particles with the
        T-Matrix method. Uses a wrapper to the Fortran code by M. Mishchenko.

        Parameters
        ----------
        izaDeg, vzaDeg, iaaDeg, vaaDeg : None, int, float or array_like
            Incidence (iza) and scattering (vza) zenith angle and incidence and viewing
            azimuth angle (ira, vra) in [DEG].
        alpha, beta: None, int, float or array_like
            The Euler angles of the particle orientation in [DEG].

        Returns
        -------
        S, Z : array_like
            Three dimensional scattering (S) and phase (Z) matrix.

        Note
        ----
        If xzaDeg, xaaDeg, alpha pr beta is None, the inputed angles in __init__ will be choose.

        !!! IMPORTANT !!!
        If the angles are NOT NONE, the new values will NOT be affect the property calls S, Z and SZ!

        """
        if izaDeg is not None:
            _, izaDeg = align_all((self.izaDeg, izaDeg), dtype=np.double)
        else:
            izaDeg = self.izaDeg

        if vzaDeg is not None:
            _, vzaDeg = align_all((self.vzaDeg, vzaDeg), dtype=np.double)
        else:
            vzaDeg = self.vzaDeg

        if iaaDeg is not None:
            _, iaaDeg = align_all((self.iaaDeg, iaaDeg), dtype=np.double)
        else:
            iaaDeg = self.iaaDeg

        if vaaDeg is not None:
            _, vaaDeg = align_all((self.vaaDeg, vaaDeg), dtype=np.double)
        else:
            vaaDeg = self.vaaDeg

        if alphaDeg is not None:
            _, alphaDeg = align_all((self.alphaDeg, alphaDeg), dtype=np.double)
        else:
            alphaDeg = self.alphaDeg

        if betaDeg is not None:
            _, betaDeg = align_all((self.betaDeg, betaDeg), dtype=np.double)
        else:
            betaDeg = self.betaDeg

        if self.orient is 'S':
            S, Z = SZ_S_VEC_WRAPPER(self.nmax, self.wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)
        elif self.orient is 'AF':
            S, Z = SZ_AF_VEC_WRAPPER(self.nmax, self.wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, self.n_alpha,
                                     self.n_beta, self.or_pdf)
        else:
            raise ValueError("Orientation must be S or AF.")

        return S, Z

    # def ifunc_Z(self, iza, iaa, vzaDeg, vaaDeg, alphaDeg, betaDeg, nmax, wavelength, pol=None):
    #     """
    #     Function to integrate the phase matrix which is compatible with scipy.integrate.dblquad.
    #
    #     Parameters
    #     ----------
    #     iza, iaa : float
    #         x and y value of integration (0, pi) and (0, 2*pi) in [RAD], respectively.
    #     vzaDeg, vaaDeg, alphaDeg, betaDeg: int, float or array_like
    #         Scattering (vza) zenith angle and viewing azimuth angle (vaa) in [DEG]. Parameter alphaDeg and betaDeg
    #         are the Euler angles of the particle orientation in [DEG].
    #     pol : int or None
    #         Polarization:
    #             * 0 : VV
    #             * 1 : HH
    #             * None (default) : Both.
    #
    #     Examples
    #     --------
    #     scipy.integrate.dblquad(TMatrix.ifunc_Z, 0, 360.0, lambda x: 0.0, lambda x: 180.0, args=(*args ))
    #
    #     Returns
    #     -------
    #     Z : float
    #
    #     """
    #     izaDeg, iaaDeg = iza * RAD_TO_DEG, iaa * RAD_TO_DEG
    #
    #     Z = self.__calc_SZ_ifunc(izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg, nmax, wavelength)[1]
    #
    #     if pol is None:
    #         return Z
    #     elif pol == 0:
    #         return Z[0, 0]
    #     elif pol == 1:
    #         return Z[1, 1]

    # ------------------------------------------------------------------------------------------------------------------
    #  Auxiliary functions and private methods
    # ------------------------------------------------------------------------------------------------------------------
    # NMAX, S and Z ----------------------------------------------------------------------------------------------------
    # ---- NMAX ----
    def __NMAX(self):
        """
        Calculate NMAX parameter.
        """
        if self.radius_type == 2:
            # Maximum radius is not directly supported in the original
            # so we convert it to equal volume radius
            radius_type = 1
            radius = np.zeros_like(self.iza)

            for i, item in enumerate(self.radius):
                radius[i] = equal_volume_from_maximum_wrapper(item, self.axis_ratio[i], self.shape)

        else:
            radius_type = self.radius_type
            radius = self.radius

        nmax = NMAX_VEC_WRAPPER(radius=radius, radius_type=radius_type, wavelength=self.wavelength,
                                eps_real=self.epsr, eps_imag=self.epsi,
                                axis_ratio=self.axis_ratio, shape=self.shape, verbose=self.verbose)

        self.radius = radius

        return nmax

    # Integration of Phase and Scattering Matrix -----------------------------------------------------------------------
    def __dblquad(self, iza_flag=True):

        if iza_flag:
            xzaDeg = self.vzaDeg
            xaaDeg = self.vaaDeg
        else:
            xzaDeg = self.izaDeg
            xaaDeg = self.iaaDeg

        if self.orient is 'S':
            Z = DBLQUAD_Z_S_WRAPPER(self.nmax, self.wavelength, xzaDeg, xaaDeg, self.alphaDeg, self.betaDeg, iza_flag)
        elif self.orient is 'AF':
            Z = DBLQUAD_Z_S_WRAPPER(self.nmax, self.wavelength, xzaDeg, xaaDeg, self.n_alpha, self.n_beta, self.or_pdf,
                                    iza_flag)
        else:
            raise ValueError("Orientation must be S or AF.")

        return Z.base.reshape((self.xmax, 4, 4))

    # Cross Section ----------------------------------------------------------------------------------------------------
    def __QS(self):
        """
        Scattering cross section for the current setup, with polarization.
        """

        if self.orient is 'S':
            QS = XSEC_QS_S_WRAPPER(self.nmax, self.wavelength, self.izaDeg, self.iaaDeg, self.alphaDeg, self.betaDeg,
                                   verbose=self.verbose)
        elif self.orient is 'AF':
            QS = XSEC_QS_S_WRAPPER(self.nmax, self.wavelength, self.izaDeg, self.iaaDeg, self.n_alpha, self.n_beta,
                                   self.or_pdf, verbose=self.verbose)
        else:
            raise ValueError("Orientation must be S or AF.")

        return QS

    def __QAS(self):
        """
        Asymetry cross section for the current setup, with polarization.
        """

        if self.orient is 'S':
            QAS = XSEC_ASY_S_WRAPPER(self.nmax, self.wavelength, self.izaDeg, self.iaaDeg, self.alphaDeg, self.betaDeg,
                                     verbose=self.verbose)
        elif self.orient is 'AF':
            QAS = XSEC_ASY_AF_WRAPPER(self.nmax, self.wavelength, self.izaDeg, self.iaaDeg, self.n_alpha, self.n_beta,
                                      self.or_pdf, verbose=self.verbose)
        else:
            raise ValueError("Orientation must be S or AF.")

        return QAS

    def __QE(self):
        """
        Extinction cross section for the current setup, with polarization.
        """

        S, Z = self.compute_SZ(vzaDeg=self.izaDeg, vaaDeg=self.iaaDeg)

        return XSEC_QE_WRAPPER(S, self.wavelength)

    def __I(self):
        """
        Scattering intensity (phase function) for the current setup.
        """
        Z = self.Z

        return XSEC_QSI_WRAPPER(Z)

    # ---- Other Functions ----
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
        elif pdf is None or pdf is 'gauss':
            return Orientation.gaussian()
        elif pdf is 'uniform':
            return Orientation.uniform()
        else:
            raise AssertionError(
                "The Particle size distribution (psd) must be callable or 'None' to get the default gaussian psd.")

    def __property_return(self, X, normalize=True):
        if normalize:
            if self.normalize:
                # return X[0:-1] - X[-1]
                return X
            else:
                return X
        else:
            if self.normalize:
                # return X[0:-1]
                return X
            else:
                return X

    # SubSection 2 -----------------------------------------------------------------------------------------------------
    # def __calc_SZ_ifunc(self, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg, nmax, wavelength):
    #
    #     if self.orient is 'S':
    #         S, Z = calc_single_wrapper(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)
    #
    #     elif self.orient is 'AA':
    #         S, Z = orient_averaged_adaptive_wrapper(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, self.or_pdf)
    #
    #
    #     elif self.orient is 'AF':
    #         S, Z = orient_averaged_fixed_wrapper(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, self.n_alpha,
    #                                              self.n_beta, self.or_pdf)
    #
    #     else:
    #         raise ValueError("Orientation must be S, AA or AF.")
    #
    #     return S, Z
