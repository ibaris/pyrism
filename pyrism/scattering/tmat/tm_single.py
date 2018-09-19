from __future__ import division

import numpy as np
from pyrism.core.tma import calc_nmax_wrapper, get_oriented_SZ, sca_xsect_wrapper, asm_wrapper, ext_xsect

from radarpy import Angles, asarrays, align_all

from .orientation import Orientation


class TMatrixSingle(Angles):
    def __init__(self, iza, vza, iaa, vaa, frequency, radius, eps, alpha=0.0, beta=0.0,
                 radius_type='REV', shape='SPH', orientation='S', axis_ratio=1.0, or_pdf=None, n_alpha=5, n_beta=10,
                 angle_unit='DEG'):

        """T-Matrix scattering from nonspherical particles.

        Class for simulating scattering from nonspherical particles with the
        T-Matrix method. Uses a wrapper to the Fortran code by M. Mishchenko.

        Usage instructions:

        First, the class should be be initialized. Any attributes (see below)
        can be passed as keyword arguments to the constructor. For example:
        sca = tmatrix.Scatterer(wavelength=2.0, m=complex(0,2))

        The properties of the scattering and the radiation should then be set
        as attributes of this object.

        The functions for computing the various scattering properties can then be
        called. The Scatterer object will automatically recompute the T-matrix
        and/or the amplitude and phase matrices when needed.

        Parameters
        ----------
        iza, vza, raa, ira, vra : int, float or ndarray
            Incidence (iza) and scattering (vza) zenith angle, incidence and viewing
            azimuth angle (ira, vra). If raa is defined, ira and vra are not mandatory.
        wavelength :
            The wavelength of incident light.
        radius :
            Equivalent radius.
        radius_type : {'EV', 'M', 'REA'}
            Specifacion of radius:
                * 'REV': radius is the equivalent volume radius (default).
                * 'M': radius is the maximum radius.
                * 'REA': radius is the equivalent area radius.
        eps :
            The complex refractive index.
        axis_ratio :
            The horizontal-to-rotational axis ratio.
        shape : {'SPH', 'CYL'}
            Shape of the scatter:
                * 'SPH' : spheroid,
                * 'CYL' : cylinders.
        alpha, beta:
            The Euler angles of the particle orientation (degrees).
        Kw_sqr :
            The squared reference water dielectric factor for computing
            radar reflectivity.
        orient : {'S', 'AA', 'AF'}
        The function to use to compute the scattering properties:
            * 'S': Single (default).
            * 'AA': Averaged Adaptive
            * 'AF': Averaged Fixed.
        or_pdf: {'gauss', 'uniform'}
            Particle orientation PDF for orientational averaging:
                * 'gauss': Use a Gaussian PDF (default).
                * 'uniform': Use a uniform PDR.
        n_alpha :
            Number of integration points in the alpha Euler angle.
        n_beta :
            Umber of integration points in the beta Euler angle.
        psd_integrator :
            Set this to a PSDIntegrator instance to enable size
            distribution integration. If this is None (default), size
            distribution integration is not used. See the PSDIntegrator
            documentation for more information.
        psd :
            Set to a callable object giving the PSD value for a given
            diameter (for example a GammaPSD instance); default None. Has no
            effect if psd_integrator is None.
        angle_unit : {'DEG', 'RAD'}, optional
            * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
            * 'RAD': All input angles (iza, vza, raa) are in [RAD].

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

        super(TMatrixSingle, self).__init__(iza=iza, vza=vza, raa=None, iaa=iaa, vaa=vaa, alpha=alpha, beta=beta,
                                            normalize=False,
                                            angle_unit=angle_unit)

        self.radius = radius
        self.radius_type = param[radius_type]

        self.wavelength = 29.9792458 / frequency
        self.axis_ratio = axis_ratio
        self.shape = param[shape]
        self.ddelt = 1e-3
        self.ndgs = 2
        self.alpha = alpha
        self.beta = beta
        self.orient = orientation

        self.or_pdf = self.__get_pdf(or_pdf)
        self.n_alpha = int(n_alpha)
        self.n_beta = int(n_beta)

        self.eps, _ = align_all((eps, self.iza))

        self.nmax = self.calc_nmax()
        self.__S, self.__Z = self.calc_SZ()

    @property
    def S(self):
        if len(self.__S) == 1:
            return self.__S[0]
        else:
            return self.__S

    @property
    def Z(self):
        if len(self.__Z) == 1:
            return self.__Z[0]
        else:
            return self.__Z

    def calc_nmax(self):
        """Initialize the T-matrix.
        """

        nmax = list()
        for i in range(len(self.izaDeg)):
            temp = calc_nmax_wrapper(self.radius[i], self.radius_type, self.wavelength[i], self.eps[i],
                                     self.axis_ratio[i], self.shape)

            nmax.append(temp)

        return np.asarray(nmax).flatten()

    def calc_xsec(self):

        ksVV, ksHH = self.sca_xsec()
        keVV, keHH = self.ext_xsec()

        kaVV = keVV - ksVV
        kaHH = keHH - ksHH
        omegaVV = ksVV / keVV
        omegaHH = ksHH / keHH

        return ksVV, kaVV, keVV, omegaVV, ksHH, kaHH, keHH, omegaHH

    def calc_SZ(self):
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

    def sca_intensity(self):
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
            VV = self.__Z[i][0, 0] + self.__Z[i][0, 1]
            HH = self.__Z[i][1, 0] + self.__Z[i][1, 1]

            VV_list.append(VV)
            HH_list.append(HH)

        return np.asarray(VV_list).flatten(), np.asarray(HH_list).flatten()

    def sca_xsec(self):
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

    def ext_xsec(self):
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

    def calc_asym(self):
        xsectVV_list = list()
        xsectHH_list = list()
        for i in range(len(self.iza)):
            xsectVV, xsectHH = asm_wrapper(self.nmax[i],
                                           self.wavelength[i], self.izaDeg[i],
                                           self.iaaDeg[i], self.alphaDeg[i], self.betaDeg[i],
                                           self.n_alpha, self.n_beta,
                                           self.or_pdf, self.orient)

            xsectVV_list.append(xsectVV)
            xsectHH_list.append(xsectHH)

        return np.asarray(xsectVV_list).flatten(), np.asarray(xsectHH_list).flatten()
