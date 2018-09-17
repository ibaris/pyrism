from __future__ import division
from datetime import datetime

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os

from os.path import expanduser

from pyrism.auxil import get_version
from radarpy import Angles
import warnings
import numpy as np
from scipy.integrate import trapz
from scipy.special import gamma
import pytmatrix.scatter as scatter
import pytmatrix.tmatrix_aux as tmatrix_aux
from .calc_sz import calc_SZ_orient, calc_nmax


class PSD(object):
    def __init__(self, n0=1.0, ilambda=1.0, rmax=None, r0=0.5, mu=0.0, normalize=True, edges=None, psd=None):
        """
        Callable class to provide different PSD functions.
        """
        self.n0 = n0
        self.ilambda = ilambda
        self.rmax = rmax
        self.r0 = r0
        self.mu = mu
        self.normalize = normalize
        self.edges = edges
        self.psd = psd

    def exponential(self, r):
        """Exponential particle size distribution (PSD).

        Method to provide an exponential PSD with the given
        parameters. The attributes can also be given as arguments to the
        constructor.

        Parameters
        ----------
        n0 :
            The intercept parameter. Default is 1.0
        ilambda :
            The inverse scale parameter. Default is 1.0
        r : int or float
            Radius of particle.
        rmax : int, float or None.
            Maximum diameter to consider. If None (default) rmax will be approximated by the PSD functions.

        Returns
        -------
        PSD :
            The PSD value for the given diameter. Returns 0 for all diameters larger than D_max.

        Note
        ----
        If rmax is None the maximum diameter will be approximated by: 11/ilambda
        """
        D = r * 2

        D_max = 11.0 / self.ilambda if self.rmax is None else self.rmax * 2

        psd = self.n0 * np.exp(-self.ilambda * D)

        if np.shape(D) == ():
            if D > D_max:
                return 0.0
        else:
            psd[D > D_max] = 0.0

        return psd

    def gamma(self, r):
        """Normalized gamma particle size distribution (PSD).

        Method to provide a normalized gamma PSD with the given
        parameters. The attributes can also be given as arguments to the
        constructor.

        The PSD form of normalized form is:
        N(D) = Nw * f(mu) * (D/D0)**mu * exp(-(3.67+mu)*D/D0)
        f(mu) = 6/(3.67**4) * (3.67+mu)**(mu+4)/Gamma(mu+4)

        The PSD form of NOT normalized form is:
        N(D) = N0 * D**mu * exp(-Lambda*D)

        Parameters
        ----------
        r0 :
            The median volume radius. Default is 0.5.
        n0 :
            The intercept parameter. Default is 1.0.
        mu :
            The shape parameter. Default is 0.0.
        ilambda :
            The inverse scale parameter. It is only necessary if normalize is False. Default is 1.0.
        normalize : bool
            If True the normalized gamma function will be calculated.
        r : int or float
            Radius of particle.
        rmax : int, float or None.
            Maximum diameter to consider. If None (default) rmax will be approximated by the PSD functions.

        Returns
        -------
        PSD :
            The PSD value for the given diameter. Returns 0 for all diameters larger than D_max.

        Note
        ----
        If rmax is None the maximum diameter will be approximated by: 3 * r0 * 2 for normlalized and 11/ilambda for
        NOT normalized case.
        """
        D = r * 2

        if self.normalize:
            nf = self.n0 * 6.0 / 3.67 ** 4 * (3.67 + self.mu) ** (self.mu + 4) / gamma(self.mu + 4)

            D0 = self.r0 * 2
            D_max = 3.0 * D0 if self.rmax is None else self.rmax * 2

            d = (D / D0)
            psd = nf * np.exp(self.mu * np.log(d) - (3.67 + self.mu) * d)
            if np.shape(D) == ():
                if (D > D_max) or (D == 0.0):
                    return 0.0
            else:
                psd[(D > D_max) | (D == 0.0)] = 0.0

        else:
            D_max = 11.0 / self.ilambda if self.rmax is None else self.rmax * 2
            psd = self.n0 * np.exp(self.mu * np.log(D) - self.ilambda * D)
            if np.shape(D) == ():
                if (D > D_max) or (D == 0):
                    return 0.0
            else:
                psd[(D > D_max) | (D == 0)] = 0.0
            return psd

        return psd

    def binned(self, r):
        """

        Parameters
        ----------
        edges : array_like
            n bin edges.
        psd : array_like
            n+1 psd values.
        r : int or float
            Radius of particle.

        Returns
        -------
        PSD :
            The PSD value for the given diameter.
            Returns 0 for all diameters outside the bins.
        """
        D = r * 2

        if len(self.edges) != len(self.psd) + 1:
            raise ValueError("There must be n+1 bin edges for n bins.")

        if not (self.edges[0] < D <= self.edges[-1]):
            return 0.0

        # binary search for the right bin
        start = 0
        end = len(self.edges)
        while end - start > 1:
            half = (start + end) // 2
            if self.edges[start] < D <= self.edges[half]:
                end = half
            else:
                start = half

        return self.psd[start]


class TMatrix_PSD(Angles):
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

    def __init__(self, iza=None, vza=None, iaa=None, vaa=None, frequency=None, radius=None, eps=None, radius_type='REV',
                 axis_ratio=1.0, shape='SPH', alpha=0.0, beta=0.0, Kw_sqr=0.93, orient='S', or_pdf='gauss', n_alpha=5,
                 n_beta=10, psd=None, num_points=1024, rmax=10, angle_unit='DEG'):

        super(TMatrix_PSD, self).__init__(iza=iza, vza=vza, raa=None, iaa=iaa, vaa=vaa, normalize=False,
                                          angle_unit=angle_unit)

        param = {'REV': 1.0,
                 'REA': 0.0,
                 'M': 2.0,
                 'SPH': -1,
                 'CYL': -2}

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
        self.Kw_sqr = Kw_sqr
        self.orient = orient

        self.or_pdf = or_pdf
        self.n_alpha = n_alpha
        self.n_beta = n_beta

        self.psd = psd

        if iza is None and vza is None and iaa is None and vaa is None:
            self.geometries = ((90.0, 90.0, 0.0, 180.0, 0.0, 0.0),)

        else:
            super(PSDIntegrator, self).__init__(iza=iza, vza=vza, raa=None, iaa=iaa, vaa=vaa, normalize=False,
                                                angle_unit=angle_unit)

            self.geometries = ((self.izaDeg, self.vzaDeg, self.iaaDeg, self.vaaDeg, 0.0, 0.0),)

        self.num_points = num_points
        self.D_max = rmax * 2

        self._S_table = None
        self._Z_table = None
        self._angular_table = None
        self._previous_psd = None

    def __trapz_sca_xsect(self, geom):
        psd_w = self.psd(self._psd_D)

        return trapz(self._angular_table["sca_xsect"][geom] * psd_w, self._psd_D)

    def sca_xsect(self):
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

        sca_prop = self.__trapz_sca_xsect(self.geometries)

        return sca_prop

    def ext_xsect(self):
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

        psd_w = self.psd(self._psd_D)

        sca_prop = trapz(self._angular_table["ext_xsect"][self.geometries] * psd_w, self._psd_D)

        return sca_prop

    def asym(self):
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

        psd_w = self.psd(self._psd_D)

        sca_xsect_int = self.__trapz_sca_xsect(self.geometries)
        if sca_xsect_int > 0:
            sca_prop = trapz(
                self._angular_table["asym"][self.geometries] * self._angular_table["sca_xsect"][
                    self.geometries] * psd_w, self._psd_D)

            sca_prop /= sca_xsect_int
        else:
            sca_prop = 0.0

        return sca_prop

    def init_scatter_table(self, angular_integration=False,
                           verbose=False):
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

        self._psd_D = np.linspace(self.D_max / self.num_points, self.D_max, self.num_points)

        self._S_table = {}
        self._Z_table = {}
        self._previous_psd = None

        self._m_table = np.empty(self.num_points, dtype=complex)

        if angular_integration:
            self._angular_table = {"sca_xsect": {}, "ext_xsect": {}, "asym": {}}
        else:
            self._angular_table = None

        for geom in self.geometries:
            self._S_table[geom] = np.empty((2, 2, self.num_points), dtype=complex)
            self._Z_table[geom] = np.empty((4, 4, self.num_points))

            if angular_integration:
                for int_var in ["sca_xsect", "ext_xsect", "asym"]:
                    self._angular_table[int_var][geom] = np.empty(self.num_points)

        for (i, D) in enumerate(self._psd_D):

            if verbose:
                print("Computing point {i} at D={D}...".format(i=i, D=D))

            m = self.eps(D) if callable(self.eps) else self.eps
            axis_ratio = self.axis_ratio(D) if callable(self.axis_ratio) else self.axis_ratio

            self._m_table[i] = m
            radius = D / 2.0

            for geom in self.geometries:
                iza, vza, iaa, vaa, alpha, beta = geom

                nmax = calc_nmax(radius, self.radius_type, self.wavelength, m, axis_ratio, self.shape)

                S, Z = calc_SZ_orient(nmax, self.wavelength, iza, vza, iaa, vaa, alpha, beta, self.n_alpha, self.n_beta,
                                      self.or_pdf,
                                      self.orient)

                self._S_table[geom][:, :, i] = S
                self._Z_table[geom][:, :, i] = Z

                if angular_integration:
                    self._angular_table["sca_xsect"][geom][i] = self.sca_xsect()
                    self._angular_table["ext_xsect"][geom][i] = self.ext_xsect()
                    self._angular_table["asym"][geom][i] = self.asym()

    def get_SZ(self, psd, geometry):
        """
        Compute the scattering matrices for the given PSD and geometries.

        Returns:
            The new amplitude (S) and phase (Z) matrices.
        """
        if (self._S_table is None) or (self._Z_table is None):
            raise AttributeError(
                "Initialize or load the scattering table first.")

        if (not isinstance(psd, PSD)) or self._previous_psd != psd:
            self._S_dict = {}
            self._Z_dict = {}

            psd_w = psd(self._psd_D)

            for geom in self.geometries:
                self._S_dict[geom] = \
                    trapz(self._S_table[geom] * psd_w, self._psd_D)
                self._Z_dict[geom] = \
                    trapz(self._Z_table[geom] * psd_w, self._psd_D)

            self._previous_psd = psd

        return self._S_dict[geometry], self._Z_dict[geometry]

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

        path = os.path.join(expanduser("~"), '.pyrism')

        if not os.path.exists(path):
            os.makedirs(path)

        else:
            pass

        if fn is None:
            time = datetime.now().isoformat()
            time = time.replace(':', '_')

            name = 'pyrism_scattering_table.p'

            fn = time + '_' + name

        files = os.path.join(path, fn)

        data = {
            "description": description,
            "time": datetime.now(),
            "psd_scatter": (self.num_points, self.D_max, self._psd_D,
                            self._S_table, self._Z_table, self._angular_table,
                            self._m_table, self.geometries),
            "version": get_version()
        }
        pickle.dump(data, open(files, 'w'), pickle.HIGHEST_PROTOCOL)


    def load_scatter_table(self, fn):
        """Load the scattering lookup tables.

        Load the scattering lookup tables saved with save_scatter_table.

        Args:
            fn: The name of the scattering table file.
        """
        data = pickle.load(file(fn))

        if ("version" not in data) or (data["version"] != tmatrix_aux.VERSION):
            warnings.warn("Loading data saved with another version.", Warning)

        (self.num_points, self.D_max, self._psd_D, self._S_table,
         self._Z_table, self._angular_table, self._m_table,
         self.geometries) = data["psd_scatter"]
        return (data["time"], data["description"])