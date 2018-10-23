from __future__ import division

import sys
import warnings

import numpy as np
from pyrism.core.tma import (calc_nmax_wrapper,
                             sca_intensity_wrapper, calc_single_wrapper,
                             orient_averaged_adaptive_wrapper, orient_averaged_fixed_wrapper,
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
from scipy.integrate import dblquad

PI = 3.14159265359
RAD_TO_DEG = 180.0 / PI
DEG_TO_RAD = PI / 180.0


class TMatrixSingle(Angles, object):
    def __init__(self, iza, vza, iaa, vaa, frequency, radius, eps, alpha=0.0, beta=0.0,
                 radius_type='REV', shape='SPH', orientation='S', axis_ratio=1.0, orientation_pdf=None, n_alpha=5,
                 n_beta=10, angle_unit='DEG', frequency_unit='GHz', normalize=False, nbar=0.0):

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

        self.wavelength = wavelength(self.frequency, unit=frequency_unit, output='cm')
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

    # ----------------------------------------------------------------------------------------------------------------------
    # Property Calls
    # ----------------------------------------------------------------------------------------------------------------------
    # ---- Scattering and Phase Matrices ----
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

    # ---- Integrated Scattering and Phase Matrices ----
    @property
    def dblquad(self):
        """
        Half space integration of the phase matrices.

        Returns
        -------
        dbl : list or array_like
        """
        try:
            if len(self.__quadZ) == 1:
                return self.__quadZ[0]
            else:
                if self.normalize:
                    if len(self.__quadZ[0:-1]) == 1:
                        return self.__quadZ[0:-1][0] - self.__quadZ[-1]
                    else:
                        return self.__quadZ[0:-1] - self.__quadZ[-1]

                else:
                    return self.__quadZ

        except AttributeError:

            self.__quadZ = self.__dblquad()

            if len(self.__quadZ) == 1:
                return self.__quadZ[0]
            else:
                if self.normalize:
                    if len(self.__quadZ[0:-1]) == 1:
                        return self.__quadZ[0:-1][0] - self.__quadZ[-1]
                    else:
                        return self.__quadZ[0:-1] - self.__quadZ[-1]

                else:
                    return self.__quadZ

    # ---- Cross Section Calls ----
    @property
    def ksx(self):
        """
        Scattering cross section for the current setup, with polarization.

        Returns
        -------
        VV, HH : list or array_like

        """
        try:
            if len(self.__ksxVV) == 1:
                return self.__ksxVV[0], self.__ksxHH[0]
            else:
                if self.normalize:
                    if len(self.__ksxVV[0:-1]) == 1:
                        return self.__ksxVV[0:-1][0], self.__ksxHH[0:-1][0]
                    else:
                        return self.__ksxVV[0:-1][0], self.__ksxHH[0:-1][0]

                else:
                    return self.__ksxVV, self.__ksxHH

        except AttributeError:

            self.__ksxVV, self.__ksxHH = self.__calc_ksx()

            if len(self.__ksxVV) == 1:
                return self.__ksxVV[0], self.__ksxHH[0]
            else:
                if self.normalize:
                    if len(self.__ksxVV[0:-1]) == 1:
                        return self.__ksxVV[0:-1][0], self.__ksxHH[0:-1][0]
                    else:
                        return self.__ksxVV[0:-1][0], self.__ksxHH[0:-1][0]

                else:
                    return self.__ksxVV, self.__ksxHH

    @property
    def kex(self):
        """
        Extinction cross section for the current setup, with polarization.

        Returns
        -------
        VV, HH : list or array_like

        """
        try:
            if len(self.__kexVV) == 1:
                return self.__kexVV[0], self.__kexHH[0]
            else:
                if self.normalize:
                    if len(self.__kexVV[0:-1]) == 1:
                        return self.__kexVV[0:-1][0], self.__kexHH[0:-1][0]
                    else:
                        return self.__kexVV[0:-1][0], self.__kexHH[0:-1][0]

                else:
                    return self.__kexVV, self.__kexHH

        except AttributeError:

            self.__kexVV, self.__kexHH = self.__calc_kex()

            if len(self.__kexVV) == 1:
                return self.__kexVV[0], self.__kexHH[0]
            else:
                if self.normalize:
                    if len(self.__kexVV[0:-1]) == 1:
                        return self.__kexVV[0:-1][0], self.__kexHH[0:-1][0]
                    else:
                        return self.__kexVV[0:-1][0], self.__kexHH[0:-1][0]

                else:
                    return self.__kexVV, self.__kexHH

    @property
    def asx(self):
        """
        Asymetry cross section for the current setup, with polarization.

        Returns
        -------
        VV, HH : list or array_like

        """
        try:
            if len(self.__asxVV) == 1:
                return self.__asxVV[0], self.__asxHH[0]
            else:
                if self.normalize:
                    if len(self.__asxVV[0:-1]) == 1:
                        return self.__asxVV[0:-1][0], self.__asxHH[0:-1][0]
                    else:
                        return self.__asxVV[0:-1][0], self.__asxHH[0:-1][0]

                else:
                    return self.__asxVV, self.__asxHH

        except AttributeError:

            self.__asxVV, self.__asxHH = self.__calc_asx()

            if len(self.__asxVV) == 1:
                return self.__asxVV[0], self.__asxHH[0]
            else:
                if self.normalize:
                    if len(self.__asxVV[0:-1]) == 1:
                        return self.__asxVV[0:-1][0], self.__asxHH[0:-1][0]
                    else:
                        return self.__asxVV[0:-1][0], self.__asxHH[0:-1][0]

                else:
                    return self.__asxVV, self.__asxHH

    @property
    def ksi(self):
        """
        Scattering intensity.

        Returns
        -------
        VV, HH : list or array_like

        """
        try:
            if len(self.__ksiVV) == 1:
                return self.__ksiVV[0], self.__ksiHH[0]
            else:
                if self.normalize:
                    if len(self.__ksiVV[0:-1]) == 1:
                        return self.__ksiVV[0:-1][0], self.__ksiHH[0:-1][0]
                    else:
                        return self.__ksiVV[0:-1][0], self.__ksiHH[0:-1][0]

                else:
                    return self.__ksiVV, self.__ksiHH

        except AttributeError:

            self.__ksiVV, self.__ksiHH = self.__calc_ksi()

            if len(self.__ksiVV) == 1:
                return self.__ksiVV[0], self.__ksiHH[0]
            else:
                if self.normalize:
                    if len(self.__ksiVV[0:-1]) == 1:
                        return self.__ksiVV[0:-1][0], self.__ksiHH[0:-1][0]
                    else:
                        return self.__ksiVV[0:-1][0], self.__ksiHH[0:-1][0]

                else:
                    return self.__ksiVV, self.__ksiHH

    # -----------------------------------------------------------------------------------------------------------------
    # User callable methods
    # -----------------------------------------------------------------------------------------------------------------
    def call_SZ(self, izaDeg=None, vzaDeg=None, iaaDeg=None, vaaDeg=None, alphaDeg=None, betaDeg=None):
        return self.__call_SZ(izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)

    def ifunc_Z(self, iza, iaa, vzaDeg, vaaDeg, alphaDeg, betaDeg, nmax, wavelength, pol=None):
        """
        Function to integrate the phase matrix which is compatible with scipy.integrate.dblquad.

        Parameters
        ----------
        iza, iaa : float
            x and y value of integration (0, pi) and (0, 2*pi) in [RAD], respectively.
        vzaDeg, vaaDeg, alphaDeg, betaDeg: int, float or array_like
            Scattering (vza) zenith angle and viewing azimuth angle (vaa) in [DEG]. Parameter alphaDeg and betaDeg
            are the Euler angles of the particle orientation in [DEG].
        pol : int or None
            Polarization:
                * 0 : VV
                * 1 : HH
                * None (default) : Both.

        Examples
        --------
        scipy.integrate.dblquad(TMatrix.ifunc_Z, 0, 360.0, lambda x: 0.0, lambda x: 180.0, args=(*args ))

        Returns
        -------
        Z : float

        """
        izaDeg, iaaDeg = iza * RAD_TO_DEG, iaa * RAD_TO_DEG

        Z = self.__calc_SZ_ifunc(izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg, nmax, wavelength)[1]

        if pol is None:
            return Z
        elif pol == 0:
            return Z[0, 0]
        elif pol == 1:
            return Z[1, 1]

    # ------------------------------------------------------------------------------------------------------------------
    #  Auxiliary functions and private methods
    # ------------------------------------------------------------------------------------------------------------------
    # ---- Scattering and Phase Matrix Calculation ----
    def __calc_nmax(self):
        """Initialize the T-matrix and calculate nmax parameter.
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

        nmax = list()

        for i in srange(len(self.izaDeg)):
            temp = calc_nmax_wrapper(radius[i], radius_type, self.wavelength[i], self.eps[i],
                                     self.axis_ratio[i], self.shape)

            nmax.append(temp)

        self.radius = radius

        return np.asarray(nmax).flatten()

    def __call_SZ(self, izaDeg=None, vzaDeg=None, iaaDeg=None, vaaDeg=None, alphaDeg=None, betaDeg=None):
        """Get the S and Z matrices for a orientated SZ.
        """

        izaDeg = self.izaDeg if izaDeg is None else izaDeg
        vzaDeg = self.vzaDeg if vzaDeg is None else vzaDeg
        iaaDeg = self.iaaDeg if iaaDeg is None else iaaDeg
        vaaDeg = self.vaaDeg if vaaDeg is None else vaaDeg
        alphaDeg = self.alphaDeg if alphaDeg is None else alphaDeg
        betaDeg = self.betaDeg if betaDeg is None else betaDeg

        S_list = list()
        Z_list = list()
        if self.orient is 'S':
            for i in srange(len(self.izaDeg)):
                S, Z = calc_single_wrapper(self.nmax[i], self.wavelength[i], izaDeg[i], vzaDeg[i],
                                           iaaDeg[i], vaaDeg[i], alphaDeg[i],
                                           betaDeg[i])

                S_list.append(S)
                Z_list.append(Z)

        elif self.orient is 'AA':
            for i in srange(len(self.izaDeg)):
                S, Z = orient_averaged_adaptive_wrapper(self.nmax[i], self.wavelength[i], izaDeg[i],
                                                        vzaDeg[i],
                                                        iaaDeg[i], vaaDeg[i], self.or_pdf)

                S_list.append(S)
                Z_list.append(Z)


        elif self.orient is 'AF':
            for i in srange(len(self.izaDeg)):
                S, Z = orient_averaged_fixed_wrapper(self.nmax[i], self.wavelength[i], izaDeg[i], vzaDeg[i],
                                                     iaaDeg[i], vaaDeg[i], self.n_alpha, self.n_beta,
                                                     self.or_pdf)

                S_list.append(S)
                Z_list.append(Z)

        else:
            raise ValueError("Orientation must be S, AA or AF.")

        return S_list, Z_list

    def __calc_SZ_ifunc(self, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg, nmax, wavelength):

        if self.orient is 'S':
            S, Z = calc_single_wrapper(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)

        elif self.orient is 'AA':
            S, Z = orient_averaged_adaptive_wrapper(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, self.or_pdf)


        elif self.orient is 'AF':
            S, Z = orient_averaged_fixed_wrapper(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, self.n_alpha,
                                                 self.n_beta, self.or_pdf)

        else:
            raise ValueError("Orientation must be S, AA or AF.")

        return S, Z

    # ---- Double Integration of Phase Matrix ----
    def __dblquad(self):
        quadZ_list = list()

        for i, geom in enumerate(self.geometriesDeg):
            izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg = geom

            quadZ = np.zeros((2, 2))

            quadZ[0, 0] = dblquad(self.ifunc_Z, 0.0, 2 * PI, lambda x: 0.0, lambda x: PI,
                                  args=(vzaDeg, vaaDeg, alphaDeg,
                                        betaDeg, self.nmax[i],
                                        self.wavelength[i],
                                        0))[0]

            quadZ[1, 1] = dblquad(self.ifunc_Z, 0.0, 2 * PI, lambda x: 0.0, lambda x: PI,
                                  args=(vzaDeg, vaaDeg, alphaDeg,
                                        betaDeg, self.nmax[i],
                                        self.wavelength[i],
                                        1))[0]

            quadZ_list.append(quadZ)

        return quadZ_list

    # ---- Cross Section Calculation ----
    def __calc_ksx(self):
        """
        Scattering cross section for the current setup, with polarization.

        Returns
        -------
        VV, HH : list or array_like
        """

        xsectVV_list = list()
        xsectHH_list = list()

        def scax_SZ(vza, vaa, func, izaDeg, iaaDeg, alphaDeg, betaDeg, nmax, wavelength, pol):
            vzaDeg = vza * RAD_TO_DEG
            vaaDeg = vaa * RAD_TO_DEG

            S, Z = func(vzaDeg=vzaDeg, vaaDeg=vaaDeg, izaDeg=izaDeg, iaaDeg=iaaDeg, alphaDeg=alphaDeg,
                        betaDeg=betaDeg, nmax=nmax, wavelength=wavelength)

            if pol == 0:
                I = Z[0, 0] + Z[0, 1]
            if pol == 1:
                I = Z[0, 0] - Z[0, 1]

            return I * np.sin(vza)

        for i in srange(len(self.iza)):
            xsectVV = \
                dblquad(scax_SZ, 0, 2 * PI, lambda x: 0.0, lambda x: PI, args=(self.__calc_SZ_ifunc, self.izaDeg[i],
                                                                               self.iaaDeg[i], self.alphaDeg[i],
                                                                               self.betaDeg[i], self.nmax[i],
                                                                               self.wavelength[i], 0))[0]

            xsectHH = \
                dblquad(scax_SZ, 0, 2 * PI, lambda x: 0.0, lambda x: PI, args=(self.__calc_SZ_ifunc, self.izaDeg[i],
                                                                               self.iaaDeg[i], self.alphaDeg[i],
                                                                               self.betaDeg[i], self.nmax[i],
                                                                               self.wavelength[i], 1))[0]

            xsectVV_list.append(xsectVV)
            xsectHH_list.append(xsectHH)

        return np.asarray(xsectVV_list).flatten(), np.asarray(xsectHH_list).flatten()

    def __calc_kex(self):
        """
        Extinction cross section for the current setup, with polarization.

        Returns
        -------
        VV, HH : list or array_like

        """

        VV_list = list()
        HH_list = list()

        S, Z = self.__call_SZ(vzaDeg=self.izaDeg, vaaDeg=self.iaaDeg)

        for i in srange(len(self.iza)):
            VV = 2 * self.wavelength[i] * S[i][0, 0].imag
            HH = 2 * self.wavelength[i] * S[i][1, 1].imag

            VV_list.append(VV)
            HH_list.append(HH)

        return np.asarray(VV_list).flatten(), np.asarray(HH_list).flatten()

    def __calc_ksi(self):
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

    def __calc_asx(self):
        """
        Asymetry factor cross section for the current setup, with polarization.

        Returns
        -------
        VV, HH : list or array_like

        """
        xsectVV_list = list()
        xsectHH_list = list()

        ksxVV, ksxHH = self.ksx

        if isinstance(ksxVV, list):
            pass
        else:
            ksxVV = [ksxVV]
            ksxHH = [ksxHH]

        def asyx_SZ(vza, vaa, func, izaDeg, iaaDeg, alphaDeg, betaDeg, nmax, wavelength, pol):
            vzaDeg = vza * RAD_TO_DEG
            vaaDeg = vaa * RAD_TO_DEG

            cos_t0 = np.cos(izaDeg * DEG_TO_RAD)
            sin_t0 = np.sin(izaDeg * DEG_TO_RAD)

            S, Z = func(vzaDeg=vzaDeg, vaaDeg=vaaDeg, izaDeg=izaDeg, iaaDeg=iaaDeg, alphaDeg=alphaDeg,
                        betaDeg=betaDeg, nmax=nmax, wavelength=wavelength)

            if pol == 0:
                I = Z[0, 0] + Z[0, 1]
            if pol == 1:
                I = Z[0, 0] - Z[0, 1]

            cos_T_sin_t = 0.5 * (
                    np.sin(2 * vza) * cos_t0 + (1 - np.cos(2 * vza)) * sin_t0 * np.cos((iaaDeg * DEG_TO_RAD) - vaa))

            return I * cos_T_sin_t

        for i in srange(len(self.iza)):
            xsectVV = \
                dblquad(asyx_SZ, 0, 2 * PI, lambda x: 0.0, lambda x: PI, args=(self.__calc_SZ_ifunc, self.izaDeg[i],
                                                                               self.iaaDeg[i], self.alphaDeg[i],
                                                                               self.betaDeg[i], self.nmax[i],
                                                                               self.wavelength[i], 0))[0]

            xsectHH = \
                dblquad(asyx_SZ, 0, 2 * PI, lambda x: 0.0, lambda x: PI, args=(self.__calc_SZ_ifunc, self.izaDeg[i],
                                                                               self.iaaDeg[i], self.alphaDeg[i],
                                                                               self.betaDeg[i], self.nmax[i],
                                                                               self.wavelength[i], 1))[0]

            xsectVV_list.append(xsectVV / ksxVV[i])
            xsectHH_list.append(xsectHH / ksxHH[i])

        return np.asarray(xsectVV_list).flatten(), np.asarray(xsectHH_list).flatten()

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
        elif pdf is None:
            return Orientation.gaussian()
        else:
            raise AssertionError(
                "The Particle size distribution (psd) must be callable or 'None' to get the default gaussian psd.")
