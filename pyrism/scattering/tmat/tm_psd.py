from __future__ import division

from datetime import datetime

from pyrism.core.tma import calc_nmax_wrapper, get_oriented_SZ, sca_xsect_wrapper, ext_xsect, asym_wrapper
from radarpy import Angles, asarrays, align_all, wavelength

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os

from pyrism.auxil import get_version, Files

import warnings

import numpy as np
from scipy.integrate import trapz
from .orientation import Orientation
from .tm_auxiliary import param

warnings.simplefilter('default')


class TMatrixPSD(Angles, object):

    def __init__(self, iza, vza, iaa, vaa, frequency, radius, eps, alpha=0.0, beta=0.0, max_radius=10,
                 radius_type='REV', shape='SPH', orientation='S', axis_ratio=1.0, orientation_pdf=None, psd=None,
                 n_alpha=5, n_beta=10, num_points=1024, angle_unit='DEG', frequency_unit='GHz',
                 angular_integration=True,
                 normalize=False, nbar=0.0):
        """T-Matrix scattering from an arrangement of nonspherical particles.

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
        psd : callable
            Particle Size Distribution Function (PSD). See pyrism.PSD.
        num_points : int
            The number of points for which to sample the PSD and
            scattering properties for; default num_points=1024 should be good
            for most purposes
        angular_integration : bool
            If True, also calculate the angle-integrated quantities (scattering cross section,
            extinction cross section, asymmetry parameter). The default is True.
         max_radius : int, float or None:
            Maximum diameter to consider. If None (default) max_radius will be approximated by the PSD functions.
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
        TMatrixPSD.save_scatter_table : None
            Save all results to a file.

        See Also
        --------
        radarpy.Angles
        pyrism.PSD

        """
        # ---- Define angles and align data ----

        n_alpha, n_beta, max_radius = asarrays((n_alpha, n_beta, max_radius))

        iza, vza, iaa, vaa, frequency, radius, axis_ratio, alpha, beta = align_all(
            (iza, vza, iaa, vaa, frequency, radius, axis_ratio, alpha, beta))

        _, eps = align_all((iza, eps))

        Angles.__init__(self, iza=iza, vza=vza, raa=None, iaa=iaa, vaa=vaa, alpha=alpha, beta=beta,
                        normalize=normalize, angle_unit=angle_unit, nbar=nbar)

        if normalize:
            _, frequency, radius, axis_ratio, alpha, beta = align_all(
                (self.iza, frequency, radius, axis_ratio, alpha, beta))

        self.angular_integration = angular_integration
        self.radius = radius
        self.radius_type = param[radius_type]
        self.frequency = frequency

        self.wavelength = wavelength(self.frequency, unit=frequency_unit, output='cm')
        self.eps = eps
        self.axis_ratio = axis_ratio
        self.shape = param[shape]
        self.ddelt = 1e-3
        self.ndgs = 2
        self.alpha = alpha
        self.beta = beta
        self.orient = orientation

        self.or_pdf = self.__get_pdf(orientation_pdf)
        self.n_alpha = n_alpha.astype(int)
        self.n_beta = n_beta.astype(int)

        self.psd = psd

        self.normalize = normalize

        self.num_points = num_points
        self.D_max = max_radius * 2

        self._S_table = None
        self._Z_table = None
        self._angular_table = None

        self._psd_D = np.linspace(self.D_max / self.num_points, self.D_max, self.num_points)

        # self.init_SZ()

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
            self.init_SZ()
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
                self.init_SZ()
                self.__S, self.__Z = self.__call_SZ()

                return self.__Z[-1]
        else:
            return None

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
            self.init_SZ()
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
    def init_SZ(self, verbose=False):
        """Initialize the scattering and phase lookup tables.

        Initialize the scattering lookup tables for the different geometries.

        Parameters
        ----------
        verbose : bool
            Print calculation steps.

        Returns
        -------
        None
        """

        self._S_table = {}
        self._Z_table = {}
        self.nmax = list()

        self._m_table = np.empty(self.num_points, dtype=complex)

        if self.angular_integration:
            self._angular_table = {"sca_xsect_VV": {}, "ext_xsect_VV": {}, "asym_VV": {}, "sca_xsect_HH": {},
                                   "ext_xsect_HH": {}, "asym_HH": {}}
        else:
            self._angular_table = None

        for geom in self.geometriesDeg:
            self._S_table[geom] = np.empty((2, 2, self.num_points), dtype=complex)
            self._Z_table[geom] = np.empty((4, 4, self.num_points))

            if self.angular_integration:
                for int_var in ["sca_xsect_VV", "ext_xsect_VV", "asym_VV", "sca_xsect_HH", "ext_xsect_HH", "asym_HH"]:
                    self._angular_table[int_var][geom] = np.empty(self.num_points)

        for (i, D) in enumerate(self._psd_D):

            if verbose:
                print("Computing point {i} at D={D}...".format(i=i, D=D))

            m = self.eps(D) if callable(self.eps) else self.eps
            axis_ratio = self.axis_ratio(D) if callable(self.axis_ratio) else self.axis_ratio

            self._m_table[i] = m
            radius = D / 2.0
            nmax = calc_nmax_wrapper(radius, self.radius_type, self.wavelength, m, axis_ratio, self.shape)
            self.nmax.append(nmax)
            for geom in self.geometriesDeg:
                iza, vza, iaa, vaa, alpha, beta = geom

                S, Z = get_oriented_SZ(nmax, self.wavelength, iza, vza, iaa, vaa, alpha, beta, self.n_alpha,
                                       self.n_beta, self.or_pdf, self.orient)

                self._S_table[geom][:, :, i] = S
                self._Z_table[geom][:, :, i] = Z

                if self.angular_integration:
                    sca_xsect_VV, sca_xsect_HH = sca_xsect_wrapper(nmax,
                                                                   self.wavelength,
                                                                   iza,
                                                                   iaa,
                                                                   alpha,
                                                                   beta,
                                                                   self.n_alpha,
                                                                   self.n_beta,
                                                                   self.or_pdf,
                                                                   self.orient)

                    ext_xsect_VV, ext_xsect_HH = ext_xsect(nmax, self.wavelength, iza, vza, iaa, vaa, alpha, beta,
                                                           self.n_alpha, self.n_beta, self.or_pdf, self.orient)

                    asym_xsect_VV, asym_xsect_HH = asym_wrapper(nmax,
                                                                self.wavelength,
                                                                iza,
                                                                iaa,
                                                                alpha,
                                                                beta,
                                                                self.n_alpha,
                                                                self.n_beta,
                                                                self.or_pdf,
                                                                self.orient)

                    self._angular_table["sca_xsect_VV"][geom][i] = sca_xsect_VV
                    self._angular_table["sca_xsect_HH"][geom][i] = sca_xsect_HH

                    self._angular_table["ext_xsect_VV"][geom][i] = ext_xsect_VV
                    self._angular_table["ext_xsect_HH"][geom][i] = ext_xsect_HH

                    self._angular_table["asym_VV"][geom][i] = asym_xsect_VV
                    self._angular_table["asym_HH"][geom][i] = asym_xsect_HH

    def ksx(self, geometries):
        """
        Scattering cross section for the current setup, with polarization.

        Parameters
        ----------
        geometries : tuple
            A tuple with (iza, vza, raa, alpha, beta) in [DEG]

        Returns
        -------
        VV, HH : list or array_like

        """
        if self._angular_table is None:
            warnings.warn("No scattering table is initialized. Try to initialize scattering table.")
            if self.angular_integration:
                self.init_SZ()
            else:
                raise AssertionError("Angular integration must be True for cross section calculation.")

        ksVV = self.__trapz_sca_xsect(geometries, 1)
        ksHH = self.__trapz_sca_xsect(geometries, 2)

        return ksVV, ksHH

    def kex(self, geometries):
        """
        Extinction cross section for the current setup, with polarization.

        Parameters
        ----------
        geometries : tuple
            A tuple with (iza, vza, raa, alpha, beta) in [DEG]

        Returns
        -------
        VV, HH : list or array_like

        """
        if self._angular_table is None:
            warnings.warn("No scattering table is initialized. Try to initialize scattering table.")
            if self.angular_integration:
                self.init_SZ()
            else:
                raise AssertionError("Angular integration must be True for cross section calculation.")

        psd_w = self.psd(self._psd_D / 2)

        keVV = trapz(self._angular_table["ext_xsect_VV"][geometries] * psd_w, self._psd_D)
        keHH = trapz(self._angular_table["ext_xsect_HH"][geometries] * psd_w, self._psd_D)

        return keVV, keHH

    def asx(self, geometries):
        """
        Asymetry factor cross section for the current setup, with polarization.

        Parameters
        ----------
        geometries : tuple
            A tuple with (iza, vza, raa, alpha, beta) in [DEG]

        Returns
        -------
        VV, HH : list or array_like

        """
        if self._angular_table is None:
            warnings.warn("No scattering table is initialized. Try to initialize scattering table.")
            if self.angular_integration:
                self.init_SZ()
            else:
                raise AssertionError("Angular integration must be True for cross section calculation.")

        psd_w = self.psd(self._psd_D / 2)

        ksVV = self.__trapz_sca_xsect(geometries, 1)
        ksHH = self.__trapz_sca_xsect(geometries, 2)

        if ksVV > 0:
            asym_VV = trapz(
                self._angular_table["asym_VV"][geometries] * self._angular_table["sca_xsect_VV"][geometries] * psd_w,
                self._psd_D)

            asym_VV /= ksVV
        else:
            asym_VV = 0.0

        if ksHH > 0:
            asym_HH = trapz(
                self._angular_table["asym_HH"][geometries] * self._angular_table["sca_xsect_HH"][geometries] * psd_w,
                self._psd_D)

            asym_HH /= ksHH
        else:
            asym_HH = 0.0

        return asym_VV, asym_HH

    # ---- Auxiliary functions and private methods ----
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

    def __trapz_sca_xsect(self, geom, pol):
        """
        Trapz integration fpr cross section.

        Parameters
        ----------
        geom : tuple
            A tuple with (iza, vza, raa, alpha, beta) in [DEG]
        pol : bool
            Polarization.
                * 0 : VV
                * 1 : HH

        Returns
        -------
        Integrated cross section.
        """
        psd_w = self.psd(self._psd_D / 2)

        if pol == 1:
            return trapz(self._angular_table["sca_xsect_VV"][geom] * psd_w, self._psd_D)
        if pol == 2:
            return trapz(self._angular_table["sca_xsect_HH"][geom] * psd_w, self._psd_D)

    def __call_SZ(self):
        """
        Compute the scattering matrices for the given PSD and geometries.

        Returns:
            The new amplitude (S) and phase (Z) matrices.
        """
        if (self._S_table is None) or (self._Z_table is None):
            raise AttributeError(
                "Initialize or load the scattering table first.")

        self._S_dict = {}
        self._Z_dict = {}

        S_list = list()
        Z_list = list()

        psd_w = self.psd(self._psd_D / 2)

        for geom in self.geometriesDeg:
            S_list.append(trapz(self._S_table[geom] * psd_w, self._psd_D))
            Z_list.append(trapz(self._Z_table[geom] * psd_w, self._psd_D))

        # return self._S_dict[geometry], self._Z_dict[geometry]
        return S_list, Z_list

    def save_scatter_table(self, fn=None, description=""):
        """Save the scattering lookup tables.

        Save the state of the scattering lookup tables to a file.
        This can be loaded later with load_scatter_table.

        Other variables will not be saved, but this does not matter because
        the results of the computations are based only on the contents
        of the table.

        Args:
           fn: The name of the scattering table file.
           description (optional): A description of the table.
        """

        if fn is None:
            files = os.path.join(Files.path, Files.generate_fn())

        else:
            files = os.path.join(Files.path, fn)

        data = {
            "description": description,
            "time": datetime.now(),
            "psd_scatter": (self.num_points,
                            self.D_max,
                            self._psd_D,
                            self._S_table,
                            self._Z_table,
                            self._angular_table,
                            self._m_table,
                            self.geometriesDeg),

            "parameter": (self.izaDeg,
                          self.vzaDeg,
                          self.iaaDeg,
                          self.vaaDeg,
                          self.angular_integration,
                          self.radius,
                          self.radius_type,

                          self.wavelength,
                          self.eps,
                          self.axis_ratio,
                          self.shape,
                          self.ddelt,
                          self.ndgs,
                          self.alpha,
                          self.beta,
                          self.orient,

                          self.or_pdf,
                          self.n_alpha,
                          self.n_beta,

                          self.psd),

            "version": get_version()
        }
        pickle.dump(data, open(files, 'w'), pickle.HIGHEST_PROTOCOL)

        Files.refresh()

    def load_scatter_table(self, fn=None):
        """Load the scattering lookup tables.

        Load the scattering lookup tables saved with save_scatter_table.

        Args:
            fn: The name of the scattering table file.
        """
        if fn is None:
            fn = Files.select_newest()
            data = pickle.load(open(os.path.join(Files.path, fn)))
        else:
            data = pickle.load(open(fn))

        if ("version" not in data) or (data["version"] != get_version()):
            warnings.warn("Loading data saved with another version.", Warning)

        (self.num_points, self.D_max, self._psd_D, self._S_table, self._Z_table, self._angular_table, self._m_table,
         self.geometriesDeg) = data["psd_scatter"]

        (self.izaDeg, self.vzaDeg, self.iaaDeg, self.vaaDeg, self.angular_integration, self.radius, self.radius_type,
         self.wavelength, self.eps, self.axis_ratio, self.shape, self.ddelt, self.ndgs, self.alpha, self.beta,
         self.orient, self.or_pdf, self.n_alpha, self.n_beta, self.psd) = data["parameter"]

        return (data["time"], data["description"])
