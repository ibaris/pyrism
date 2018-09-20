from __future__ import division

from pyrism.core.tma import calc_nmax_wrapper, get_oriented_SZ, sca_xsect_wrapper, ext_xsect, asym_wrapper

from datetime import datetime

from radarpy import Angles, asarrays

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


class TMatrixPSD(Angles):
    """A class used to perform computations over PSDs.

    This class can be used to integrate scattering properties over particle
    size distributions.

    Initialize an instance of the class and set the attributes as described
    below. Call init_scatter_table to compute the lookup table for scattering
    values at different scatterer geometries. Set the class instance as the
    psd_integrator attribute of a Scatterer object to enable PSD averaging for
    that object.

    After a call to init_scatter_table, the scattering properties can be
    retrieved multiple times without re-initializing. However, the geometry of
    the Scatterer instance must be set to one of those specified in the
    "geometries" attribute.

    Attributes:

        num_points: the number of points for which to sample the PSD and
            scattering properties for; default num_points=1024 should be good
            for most purposes
        m_func: set to a callable object giving the refractive index as a
            function of diameter, or None to use the "m" attribute of the
            Scatterer for all sizes; default None
        axis_ratio_func: set to a callable object giving the aspect ratio
            (horizontal to rotational) as a function of diameter, or None to
            use the "axis_ratio" attribute for all sizes; default None
        D_max: set to the maximum single scatterer size that is desired to be
            used (usually the D_max corresponding to the largest PSD you
            intend to use)
        geometries: tuple containing the scattering geometry tuples that are
            initialized (thet0, thet, phi0, phi, alpha, beta);
            default horizontal backscatter
    """

    def __init__(self, iza, vza, iaa, vaa, frequency, radius, eps, alpha=0.0, beta=0.0, max_radius=10,
                 radius_type='REV', shape='SPH', orientation='S', axis_ratio=1.0, or_pdf=None, psd=None,
                 n_alpha=5, n_beta=10, num_points=1024,
                 angle_unit='DEG', angular_integration=True):

        iza, vza, iaa, vaa, frequency, radius, axis_ratio, alpha, beta, n_alpha, n_beta, max_radius = asarrays(
            (iza, vza, iaa, vaa, frequency, radius, axis_ratio, alpha, beta, n_alpha, n_beta, max_radius))

        eps = np.asarray(eps).flatten()

        param = {'REV': 1.0,
                 'REA': 0.0,
                 'M': 2.0,
                 'SPH': -1,
                 'CYL': -2}

        self.angular_integration = angular_integration
        self.radius = radius
        self.radius_type = param[radius_type]

        self.wavelength = 29.9792458 / frequency
        self.eps = eps
        self.axis_ratio = axis_ratio
        self.shape = param[shape]
        self.ddelt = 1e-3
        self.ndgs = 2
        self.alpha = alpha
        self.beta = beta
        self.orient = orientation

        self.or_pdf = self.__get_pdf(or_pdf)
        self.n_alpha = n_alpha.astype(int)
        self.n_beta = n_beta.astype(int)

        self.psd = psd

        super(TMatrixPSD, self).__init__(iza=iza, vza=vza, iaa=iaa, vaa=vaa, normalize=False,
                                         angle_unit=angle_unit)

        self.num_points = num_points
        self.D_max = max_radius * 2

        self._S_table = None
        self._Z_table = None
        self._angular_table = None

        self._psd_D = np.linspace(self.D_max / self.num_points, self.D_max, self.num_points)

        self.init_scatter_table()
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

    def __get_pdf(self, pdf):
        if callable(pdf):
            return pdf
        elif pdf is None:
            return Orientation.gaussian()
        else:
            raise AssertionError(
                "The Particle size distribution (psd) must be callable or 'None' to get the default gaussian psd.")

    def __trapz_sca_xsect(self, geom, pol):
        psd_w = self.psd(self._psd_D / 2)

        if pol == 1:
            return trapz(self._angular_table["sca_xsect_VV"][geom] * psd_w, self._psd_D)
        if pol == 2:
            return trapz(self._angular_table["sca_xsect_HH"][geom] * psd_w, self._psd_D)

    def calc_xsec(self):
        ksVV_list = list()
        keVV_list = list()
        ksHH_list = list()
        keHH_list = list()

        for geom in self.geometriesDeg:
            ksVV, ksHH = self.sca_xsect(geom)
            keVV, keHH = self.__ext_xsect(geom)

            ksVV_list.append(ksVV)
            keVV_list.append(keVV)
            ksHH_list.append(ksHH)
            keHH_list.append(keHH)

        ksVV = np.asarray(ksVV_list).flatten()
        ksHH = np.asarray(ksHH_list).flatten()

        keVV = np.asarray(keVV_list).flatten()
        keHH = np.asarray(keHH_list).flatten()

        kaVV = keVV - ksVV
        kaHH = keHH - ksHH

        omegaVV = ksVV / keVV
        omegaHH = ksHH / keHH

        return ksVV, kaVV, keVV, omegaVV, ksHH, kaHH, keHH, omegaHH

    def sca_xsect(self, geometries):
        """Scattering cross section for the current setup, with polarization.

        Args:
            scatterer: a Scatterer instance.
            h_pol: If True (default), use horizontal polarization.
            If False, use vertical polarization.

        Returns:
            The scattering cross section.
        """
        if self._angular_table is None:
            raise AttributeError(
                "Initialize or load the table of angular-integrated " +
                "quantities first."
            )

        ksVV = self.__trapz_sca_xsect(geometries, 1)
        ksHH = self.__trapz_sca_xsect(geometries, 2)

        return ksVV, ksHH

    def __ext_xsect(self, geometries):
        """Extinction cross section for the current setup, with polarization.

        Args:
            scatterer: a Scatterer instance.
            h_pol: If True (default), use horizontal polarization.
            If False, use vertical polarization.

        Returns:
            The extinction cross section.
        """
        if self._angular_table is None:
            raise AttributeError(
                "Initialize or load the table of angular-integrated " +
                "quantities first."
            )

        psd_w = self.psd(self._psd_D / 2)

        keVV = trapz(self._angular_table["ext_xsect_VV"][geometries] * psd_w, self._psd_D)
        keHH = trapz(self._angular_table["ext_xsect_HH"][geometries] * psd_w, self._psd_D)

        return keVV, keHH

    def __asym(self, geometries):
        """Asymmetry parameter for the current setup, with polarization.

        Args:
            scatterer: a Scatterer instance.
            h_pol: If True (default), use horizontal polarization.
            If False, use vertical polarization.

        Returns:
            The asymmetry parameter.
        """
        if self._angular_table is None:
            raise AttributeError(
                "Initialize or load the table of angular-integrated " +
                "quantities first."
            )

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

    def init_scatter_table(self, verbose=False):
        """Initialize the scattering lookup tables.

        Initialize the scattering lookup tables for the different geometries.
        Before calling this, the following attributes must be set:
           num_points, m_func, axis_ratio_func, D_max, geometries
        and additionally, all the desired attributes of the Scatterer class
        (e.g. wavelength, aspect ratio).

        Args:
            tm: a Scatterer instance.
            angular_integration: If True, also calculate the
                angle-integrated quantities (scattering cross section,
                extinction cross section, asymmetry parameter). These are
                needed to call the corresponding functions in the scatter
                module when PSD integration is active. The default is False.
            verbose: if True, print information about the progress of the
                calculation (which may take a while). If False (default),
                run silently.
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

                # if self.angular_integration:
                #     self._angular_table["sca_xsect"][geom][i] = self.sca_xsect(geom)
                #     self._angular_table["ext_xsect"][geom][i] = self.__ext_xsect(geom)
                #     self._angular_table["asym"][geom][i] = self.__asym(geom)

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

    def calc_SZ(self):
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
            "psd_scatter": (self.num_points, self.D_max, self._psd_D,
                            self._S_table, self._Z_table, self._angular_table,
                            self._m_table, self.geometriesDeg),
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

        (self.num_points, self.D_max, self._psd_D, self._S_table,
         self._Z_table, self._angular_table, self._m_table,
         self.geometriesDeg) = data["psd_scatter"]
        return (data["time"], data["description"])
