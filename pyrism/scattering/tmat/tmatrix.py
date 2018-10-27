# import numpy as np
from pyrism.core.tma import KE_WRAPPER, KS_WRAPPER, KA_WRAPPER, KT_WRAPPER
from radarpy import wavenumber

from tm_single import TMatrixSingle

PI = 3.14159265359


class TMatrix(TMatrixSingle):

    def __init__(self, iza, vza, iaa, vaa, frequency, radius, eps, alpha=0.0, beta=0.0, N=1,
                 radius_type='REV', shape='SPH', orientation='S', orientation_pdf=None, axis_ratio=1.0,
                 n_alpha=5, n_beta=10, normalize=False, nbar=0.0, angle_unit='DEG', frequency_unit='GHz',
                 radius_unit='m'):

        """T-Matrix scattering from nonspherical particles.

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
        eps : complex
            The complex refractive index.
        alpha, beta: int, float or array_like
            The Euler angles of the particle orientation in [DEG] or [RAD] (see parameter angle_unit). Default is 0.0.
        radius_type : {'EV', 'M', 'REA'}
            Specification of particle radius:
                * 'REV': radius is the equivalent volume radius (default).
                * 'M': radius is the maximum radius.
                * 'REA': radius is the equivalent area radius.
        shape : {'SPH', 'CYL'}
            Shape of the particle:
                * 'SPH' : spheroid,
                * 'CYL' : cylinders.
        orientation : {'S', 'AA', 'AF'}
            The function to use to compute the orientational scattering properties:
                * 'S': Single (default).
                * 'AA': Averaged Adaptive
                * 'AF': Averaged Fixed.
        orientation_pdf: {'gauss', 'uniform'}, callable
            Particle orientation Probability Density Function (PDF) for orientational averaging:
                * 'gauss': Use a Gaussian PDF (default).
                * 'uniform': Use a uniform PDR.
        axis_ratio : int or float
            The horizontal-to-rotational axis ratio.
        n_alpha : int
            Number of integration points in the alpha Euler angle. Default is 5.
        n_beta : int
            Umber of integration points in the beta Euler angle. Default is 10.
        num_points : int
            The number of points for which to sample the PSD and
            scattering properties for; default num_points=1024 should be good
            for most purposes.
        normalize : boolean, optional
            Set to 'True' to make kernels 0 at nadir view illumination. Since all implemented kernels are normalized
            the default value is False.
        nbar : float, optional
            The sun or incidence zenith angle at which the isotropic term is set
            to if normalize is True. The default value is 0.0.
        angle_unit : {'DEG', 'RAD'}, optional
            * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
            * 'RAD': All input angles (iza, vza, raa) are in [RAD].
        frequency_unit : {'Hz', 'MHz', 'GHz', 'THz'}
            Unit of entered frequency. Default is 'GHz'.
        radius_unit : {'m', 'cm', 'nm'}
            Unit of the radius in meter (m), centimeter (cm) or nanometer (nm).

        Returns
        -------
        TMatrix.S : array_like
            Complex Scattering Matrix.
        TMatrix.Z : array_like
            Phase Matrix.
        TMatrix.SZ : list or array_like
             Complex Scattering Matrix and Phase Matrix.

        TMatrix.ks : list or array_like
            Scattering coefficient matrix in [1/cm] for VV and HH polarization.
        TMatrix.ka : list or array_like
            Absorption coefficient matrix in [1/cm] for VV and HH polarization.
        TMatrix.ke : list or array_like
            Extinction coefficient matrix in [1/cm] for VV and HH polarization.
        TMatrix.kt : list or array_like
            Transmittance coefficient matrix in [1/cm] for VV and HH polarization.
        TMatrix.omega : list or array_like
            Single scattering albedo coefficient matrix in [1/cm] for VV and HH polarization.

        TMatrix.QS : list or array_like
            Scattering Cross Section in [cm^2] for VV and HH polarization.
        TMatrix.QE : list or array_like
            Extinction Cross Section in [cm^2] for VV and HH polarization.
        TMatrix.QAS : list or array_like
            Asymetry Factor in [cm^2] for VV and HH polarization.
        TMatrix.I : list or array_like
            Scattering intensity for VV and HH polarization.

        TMatrix.compute_SZ(...) :
            Function to recalculate SZ for different angles.

        See Also
        --------
        radarpy.Angles
        pyrism.Orientation

        """

        super(TMatrix, self).__init__(iza=iza, vza=vza, iaa=iaa, vaa=vaa, frequency=frequency, radius=radius,
                                      eps=eps,
                                      alpha=alpha, beta=beta, radius_type=radius_type, shape=shape,
                                      orientation=orientation, axis_ratio=axis_ratio, orientation_pdf=orientation_pdf,
                                      n_alpha=n_alpha,
                                      n_beta=n_beta, angle_unit=angle_unit, frequency_unit=frequency_unit,
                                      normalize=normalize, nbar=nbar, radius_unit=radius_unit)

        # --------------------------------------------------------------------------------------------------------------
        # Calculation
        # --------------------------------------------------------------------------------------------------------------
        self.k0 = wavenumber(self.frequency, unit=self.frequency_unit, output=self.radius_unit)
        self.a = self.k0 * self.radius
        self.factor = complex(0, 2 * PI * N) / self.k0


    # ------------------------------------------------------------------------------------------------------------------
    # Property Calls
    # ------------------------------------------------------------------------------------------------------------------
    # Extinction and Scattering Matrices ------------------------------------------------------------------------------

    @property
    def ke(self):
        """
        Extinction matrix for the current setup, with polarization.

        Returns
        -------
        ke : MemoryView, double[:,:}
            MemoryView of type array([[VV, HH]])
        """
        try:
            return self.__property_return(self.__ke)

        except AttributeError:
            self.__ke = self.__KE()

            return self.__property_return(self.__ke)

    @property
    def ks(self):
        """
        Scattering matrix for the current setup, with polarization.

        Returns
        -------
        ks : MemoryView, double[:,:}
            MemoryView of type array([[VV, HH]])
        """
        try:
            return self.__property_return(self.__ks)

        except AttributeError:
            self.__ks = self.__KS()

            return self.__property_return(self.__ks)

    @property
    def ka(self):
        """
        Absorption matrix for the current setup, with polarization.

        Returns
        -------
        ka : MemoryView, double[:,:}
            MemoryView of type array([[VV, HH]])
        """
        try:
            return self.__property_return(self.__ka)

        except AttributeError:
            self.__ka = self.__KA()

            return self.__property_return(self.__ka)

    @property
    def omega(self):
        """
        Single scattering albedo matrix for the current setup, with polarization.

        Returns
        -------
        omega : MemoryView, double[:,:}
            MemoryView of type array([[VV, HH]])
        """
        try:
            return self.__property_return(self.__omega)

        except AttributeError:
            self.__omega = self.__OMEGA()

            return self.__property_return(self.__omega)

    @property
    def kt(self):
        """
        Transmission matrix for the current setup, with polarization.

        Returns
        -------
        kt : MemoryView, double[:,:}
            MemoryView of type array([[VV, HH]])
        """
        try:
            return self.__property_return(self.__kt)

        except AttributeError:
            self.__kt = self.__KT()

            return self.__property_return(self.__kt)

    # ------------------------------------------------------------------------------------------------------------------
    # Auxiliary functions and private methods
    # ------------------------------------------------------------------------------------------------------------------
    # Functions to calculate the Cross Section ------------------------------------------------------------------------
    def __KT(self):
        ke = self.ke

        return KT_WRAPPER(ke)

    def __KS(self):

        ke = self.ke
        omega = self.omega

        return KS_WRAPPER(ke, omega)

    def __KA(self):
        ks = self.ks
        omega = self.omega

        return KA_WRAPPER(ks, omega)

    def __KE(self):
        S = self.S
        return KE_WRAPPER(self.factor, S)

    def __OMEGA(self):
        QS = self.QS
        QE = self.QE

        return QS.base / QE.base

    def __property_return(self, X):
        if self.normalize:
            return X[0:-1]
        else:
            return X
