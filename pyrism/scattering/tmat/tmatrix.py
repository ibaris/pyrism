from .tm_single import TMatrixSingle
from .tm_psd import TMatrixPSD
import numpy as np
from radarpy import Angles, asarrays, align_all

PI = 3.14159265359


class TMatrix(Angles):

    def __init__(self, iza, vza, iaa, vaa, frequency, radius, eps, alpha=0.0, beta=0.0,
                 radius_type='REV', shape='SPH', orientation='S', axis_ratio=1.0, orientation_pdf=None, n_alpha=5,
                 n_beta=10,
                 angle_unit='DEG', psd=None, max_radius=10, num_points=1024, angular_integration=True,
                 N=1):
        """T-Matrix scattering from nonspherical particles.

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
        psd : callable or None
            Particle Size Distribution Function (PSD). See pyrism.PSD. If None (default) a particle distribution
        num_points : int
            The number of points for which to sample the PSD and
            scattering properties for; default num_points=1024 should be good
            for most purposes
        angular_integration : bool
            If True, also calculate the angle-integrated quantities (scattering cross section,
            extinction cross section, asymmetry parameter). The default is True.
         max_radius : int, float or None:
            Maximum diameter to consider. If None (default) max_radius will be approximated by the PSD functions.
        N : int
            Amount of scatterer. Default is 1.

        Returns
        -------
        TMatrix.S : array_like
            Complex Scattering Matrix.
        TMatrix.Z : array_like
            Phase Matrix.
        TMatrix.SZ : tuple
             Complex Scattering Matrix and Phase Matrix.
        TMatrix.ksi : tuple
            Scattering intensity for VV and HH polarization.
        TMatrix.ksx : tuple
            Scattering Cross Section for VV and HH polarization.
        TMatrix.kex : tuple
            Extinction Cross Section for VV and HH polarization.
        TMatrix.asx : tuple
            Asymetry Factor for VV and HH polarization.
        TMatrix.save_scatter_table : None
            Save all results to a file. This only works if psd is defined.

        See Also
        --------
        radarpy.Angles
        pyrism.PSD

        """

        if psd is None:
            self.TM = TMatrixSingle(iza=iza, vza=vza, iaa=iaa, vaa=vaa, frequency=frequency, radius=radius,
                                    eps=eps,
                                    alpha=alpha, beta=beta, radius_type=radius_type, shape=shape,
                                    orientation=orientation, axis_ratio=axis_ratio, orientation_pdf=orientation_pdf,
                                    n_alpha=n_alpha,
                                    n_beta=n_beta, angle_unit=angle_unit)
            self.psd = None
            self.__NAME = 'SINGLE'

        else:
            self.TM = TMatrixPSD(iza=iza, vza=vza, iaa=iaa, vaa=vaa, frequency=frequency, radius=radius,
                                 eps=eps,
                                 alpha=alpha, beta=beta, radius_type=radius_type, shape=shape,
                                 orientation=orientation, axis_ratio=axis_ratio, orientation_pdf=orientation_pdf,
                                 n_alpha=n_alpha,
                                 n_beta=n_beta, angle_unit=angle_unit,

                                 psd=psd, num_points=num_points, angular_integration=angular_integration,
                                 max_radius=max_radius)

            self.__NAME = 'PSD'

            self.psd = self.TM.psd
            self.num_points = self.TM.num_points
            self.D_max = self.TM.D_max
            self.angular_integration = self.TM.angular_integration

            self._S_table = self.TM._S_table
            self._Z_table = self.TM._Z_table
            self._angular_table = self.TM._angular_table

            self._psd_D = self.TM._psd_D

        iza, vza, iaa, vaa = asarrays((iza, vza, iaa, vaa))

        iza, vza, iaa, vaa = align_all((iza, vza, iaa, vaa))

        Angles.__init__(self, iza=iza, vza=vza, raa=None, iaa=iaa, vaa=vaa, alpha=alpha, beta=beta,
                        normalize=False, angle_unit=angle_unit)

        self.radius = self.TM.radius
        self.radius_type = self.TM.radius_type

        self.wavelength = self.TM.wavelength
        self.eps = self.TM.eps
        self.axis_ratio = self.TM.axis_ratio
        self.shape = self.TM.shape
        self.ddelt = self.TM.ddelt
        self.ndgs = self.TM.ndgs
        self.alpha = self.TM.alpha
        self.beta = self.TM.beta
        self.orient = self.TM.orient

        self.or_pdf = self.TM.or_pdf
        self.n_alpha = self.TM.n_alpha
        self.n_beta = self.TM.n_beta

        self.N = N
        self.k0 = (2 * PI) / self.wavelength
        self.a = self.k0 * radius
        self.factor = complex(0, 2 * PI * self.N) / self.k0

        self.__kex = None
        self.__ksx = None
        self.__asx = None
        self.__ksi = None

    @property
    def S(self):
        return self.TM.S

    @property
    def Z(self):
        return self.TM.Z

    @property
    def SZ(self):
        return self.TM.SZ

    @property
    def ke(self):
        try:
            if len(self.__ke) == 1:
                return self.__ke[0]
            else:
                return self.__ke

        except AttributeError:

            self.__ke = self.__get_ke()

            if len(self.__ke) == 1:
                return self.__ke[0]
            else:
                return self.__ke

    @property
    def kex(self):
        if self.__kex is None:
            V, H = self.__get_kex()

            if len(V) == 1:
                return V[0], H[0]
        else:
            return self.__kex

    @property
    def ksx(self):
        if self.__ksx is None:
            V, H = self.__get_ksx()

            if len(V) == 1:
                return V[0], H[0]

        else:
            return self.__ksx

    @property
    def ksi(self):
        if self.__ksi is None:
            try:
                V, H = self.__get_ksi()

                if len(V) == 1:
                    return V[0], H[0]

            except TypeError:
                return None
        else:
            return self.__ksi

    @property
    def asx(self):
        if self.__asx is None:
            V, H = self.__get_asx()

            if len(V) == 1:
                return V[0], H[0]
        else:
            return self.__asx

    def Mpq(self, factor, S):
        return factor * S

    def __get_ke(self):
        self.__ke = list()

        if isinstance(self.TM.S, list):

            for i in range(len(self.TM.S)):
                kem = np.zeros((4, 4))

                kem[0, 0] = -2 * self.Mpq(self.factor[i], self.TM.S[i][0, 0]).real
                kem[0, 1] = 0
                kem[0, 2] = -self.Mpq(self.factor[i], self.TM.S[i][0, 1]).real
                kem[0, 3] = -self.Mpq(self.factor[i], self.TM.S[i][0, 0]).imag

                kem[1, 0] = 0
                kem[1, 1] = -2 * self.Mpq(self.factor[i], self.TM.S[i][1, 1]).real
                kem[1, 2] = -self.Mpq(self.factor[i], self.TM.S[i][1, 0]).real
                kem[1, 3] = -self.Mpq(self.factor[i], self.TM.S[i][1, 0]).imag

                kem[2, 0] = -2 * self.Mpq(self.factor[i], self.TM.S[i][1, 0]).real
                kem[2, 1] = -2 * self.Mpq(self.factor[i], self.TM.S[i][0, 1]).real
                kem[2, 2] = -(self.Mpq(self.factor[i], self.TM.S[i][0, 0]).real + self.Mpq(self.factor[i],
                                                                                           self.TM.S[i][1, 1]).real)
                kem[2, 3] = (self.Mpq(self.factor[i], self.TM.S[i][0, 0]).imag - self.Mpq(self.factor[i],
                                                                                          self.TM.S[i][1, 1]).imag)

                kem[3, 0] = 2 * self.Mpq(self.factor[i], self.TM.S[i][1, 0]).imag
                kem[3, 1] = -2 * self.Mpq(self.factor[i], self.TM.S[i][0, 1]).imag
                kem[3, 2] = -(self.Mpq(self.factor[i], self.TM.S[i][0, 0]).imag - self.Mpq(self.factor[i],
                                                                                           self.TM.S[i][1, 1]).imag)
                kem[3, 3] = -(self.Mpq(self.factor[i], self.TM.S[i][0, 0]).real + self.Mpq(self.factor[i],
                                                                                           self.TM.S[i][1, 1]).real)

                self.__ke.append(kem)

        else:
            self.__ke = list()
            kem = np.zeros((4, 4))

            kem[0, 0] = -2 * self.Mpq(self.factor, self.TM.S[0, 0]).real
            kem[0, 1] = 0
            kem[0, 2] = -self.Mpq(self.factor, self.TM.S[0, 1]).real
            kem[0, 3] = -self.Mpq(self.factor, self.TM.S[0, 0]).imag

            kem[1, 0] = 0
            kem[1, 1] = -2 * self.Mpq(self.factor, self.TM.S[1, 1]).real
            kem[1, 2] = -self.Mpq(self.factor, self.TM.S[1, 0]).real
            kem[1, 3] = -self.Mpq(self.factor, self.TM.S[1, 0]).imag

            kem[2, 0] = -2 * self.Mpq(self.factor, self.TM.S[1, 0]).real
            kem[2, 1] = -2 * self.Mpq(self.factor, self.TM.S[0, 1]).real
            kem[2, 2] = -(self.Mpq(self.factor, self.TM.S[0, 0]).real + self.Mpq(self.factor, self.TM.S[1, 1]).real)
            kem[2, 3] = (self.Mpq(self.factor, self.TM.S[0, 0]).imag - self.Mpq(self.factor, self.TM.S[1, 1]).imag)

            kem[3, 0] = 2 * self.Mpq(self.factor, self.TM.S[1, 0]).imag
            kem[3, 1] = -2 * self.Mpq(self.factor, self.TM.S[0, 1]).imag
            kem[3, 2] = -(self.Mpq(self.factor, self.TM.S[0, 0]).imag - self.Mpq(self.factor, self.TM.S[1, 1]).imag)
            kem[3, 3] = -(self.Mpq(self.factor, self.TM.S[0, 0]).real + self.Mpq(self.factor, self.TM.S[1, 1]).real)

            self.__ke.append(kem)

        return self.__ke

    def __get_asx(self):

        if self.__NAME is 'SINGLE':
            return self.TM.asx()

        else:
            try:
                if isinstance(self.geometriesDeg[0], tuple):
                    VV, HH = list(), list()

                    for item in self.geometriesDeg:
                        VV_temp, HH_temp = self.TM.asx(item)

                        VV.append(VV_temp)
                        HH.append(HH_temp)

                    return VV, HH

                else:
                    return self.TM.asx(self.geometriesDeg)

            except AttributeError:
                print("Initialising angular-integrated quantities first.")
                self.TM.init_scatter_table()

                if isinstance(self.geometriesDeg[0], tuple):
                    VV, HH = list(), list()

                    for item in self.geometriesDeg:
                        VV_temp, HH_temp = self.TM.asx(item)

                        VV.append(VV_temp)
                        HH.append(HH_temp)

                    return VV, HH

                else:
                    return self.TM.asx(self.geometriesDeg)

    def __get_ksx(self):
        if self.__NAME is 'SINGLE':
            return self.TM.ksx()

        else:
            try:
                if isinstance(self.geometriesDeg[0], tuple):
                    VV, HH = list(), list()

                    for item in self.geometriesDeg:
                        VV_temp, HH_temp = self.TM.ksx(item)

                        VV.append(VV_temp)
                        HH.append(HH_temp)

                    return VV, HH

                else:
                    return self.TM.ksx(self.geometriesDeg)

            except AttributeError:
                print("Initialising angular-integrated quantities first.")
                self.TM.init_scatter_table()

                if isinstance(self.geometriesDeg[0], tuple):
                    VV, HH = list(), list()

                    for item in self.geometriesDeg:
                        VV_temp, HH_temp = self.TM.ksx(item)

                        VV.append(VV_temp)
                        HH.append(HH_temp)

                    return VV, HH

                else:
                    return self.TM.ksx(self.geometriesDeg)

    def __get_kex(self):
        if self.__NAME is 'SINGLE':
            return self.TM.kex()

        else:
            try:
                if isinstance(self.geometriesDeg[0], tuple):
                    VV, HH = list(), list()

                    for item in self.geometriesDeg:
                        VV_temp, HH_temp = self.TM.kex(item)

                        VV.append(VV_temp)
                        HH.append(HH_temp)

                    return VV, HH

                else:
                    return self.TM.kex(self.geometriesDeg)

            except AttributeError:
                print("Initialising angular-integrated quantities first.")
                self.TM.init_scatter_table()

                if isinstance(self.geometriesDeg[0], tuple):
                    VV, HH = list(), list()

                    for item in self.geometriesDeg:
                        VV_temp, HH_temp = self.TM.kex(item)

                        VV.append(VV_temp)
                        HH.append(HH_temp)

                    return VV, HH

                else:
                    return self.TM.kex(self.geometriesDeg)

    def __get_ksi(self):
        if self.__NAME is 'SINGLE':
            return self.TM.ksi()

        else:
            return None

    @classmethod
    def load_scatter_table(cls, fn=None):
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

        (cls.num_points, cls.D_max, cls._psd_D, cls._S_table, cls._Z_table, cls._angular_table, cls._m_table,
         cls.geometriesDeg) = data["psd_scatter"]

        (cls.izaDeg, cls.vzaDeg, cls.iaaDeg, cls.vaaDeg, cls.angular_integration, cls.radius, cls.radius_type,
         cls.wavelength, cls.eps, cls.axis_ratio, cls.shape, cls.ddelt, cls.ndgs, cls.alpha, cls.beta, cls.orient,
         cls.or_pdf, cls.n_alpha, cls.n_beta, cls.psd) = data["parameter"]

        print("File {0} load successfully.".format(str(fn)))

        return (data["time"], data["description"])
