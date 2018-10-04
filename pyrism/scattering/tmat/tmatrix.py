from .tm_single import TMatrixSingle
from .tm_psd import TMatrixPSD
import numpy as np

PI = 3.14159265359


class TMatrix(TMatrixSingle, TMatrixPSD):

    def __init__(self, iza, vza, iaa, vaa, frequency, radius, eps, alpha=0.0, beta=0.0,
                 radius_type='REV', shape='SPH', orientation='AF', axis_ratio=1.0, orientation_pdf=None, n_alpha=5,
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
            TMatrixSingle.__init__(self, iza=iza, vza=vza, iaa=iaa, vaa=vaa, frequency=frequency, radius=radius,
                                   eps=eps,
                                   alpha=alpha, beta=beta, radius_type=radius_type, shape=shape,
                                   orientation=orientation, axis_ratio=axis_ratio, orientation_pdf=orientation_pdf,
                                   n_alpha=n_alpha,
                                   n_beta=n_beta, angle_unit=angle_unit)

        else:
            TMatrixPSD.__init__(self, iza=iza, vza=vza, iaa=iaa, vaa=vaa, frequency=frequency, radius=radius, eps=eps,
                                alpha=alpha, beta=beta, radius_type=radius_type, shape=shape,
                                orientation=orientation, axis_ratio=axis_ratio, orientation_pdf=orientation_pdf,
                                n_alpha=n_alpha,
                                n_beta=n_beta, angle_unit=angle_unit,

                                psd=psd, num_points=num_points, angular_integration=angular_integration,
                                max_radius=max_radius)

        self.N = N
        self.k0 = (2 * PI) / self.wavelength
        self.a = self.k0 * radius
        self.factor = complex(0, 2 * PI * self.N) / self.k0

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

    def Mpq(self, factor, S):
        return factor * S

    def __get_ke(self):
        self.__ke = list()

        if isinstance(self.S, list):

            for i in range(len(self.S)):
                kem = np.zeros((4, 4))

                kem[0, 0] = -2 * self.Mpq(self.factor[i], self.S[i][0, 0]).real
                kem[0, 1] = 0
                kem[0, 2] = -self.Mpq(self.factor[i], self.S[i][0, 1]).real
                kem[0, 3] = -self.Mpq(self.factor[i], self.S[i][0, 0]).imag

                kem[1, 0] = 0
                kem[1, 1] = -2 * self.Mpq(self.factor[i], self.S[i][1, 1]).real
                kem[1, 2] = -self.Mpq(self.factor[i], self.S[i][1, 0]).real
                kem[1, 3] = -self.Mpq(self.factor[i], self.S[i][1, 0]).imag

                kem[2, 0] = -2 * self.Mpq(self.factor[i], self.S[i][1, 0]).real
                kem[2, 1] = -2 * self.Mpq(self.factor[i], self.S[i][0, 1]).real
                kem[2, 2] = -(self.Mpq(self.factor[i], self.S[i][0, 0]).real + self.Mpq(self.factor[i],
                                                                                        self.S[i][1, 1]).real)
                kem[2, 3] = (self.Mpq(self.factor[i], self.S[i][0, 0]).imag - self.Mpq(self.factor[i],
                                                                                       self.S[i][1, 1]).imag)

                kem[3, 0] = 2 * self.Mpq(self.factor[i], self.S[i][1, 0]).imag
                kem[3, 1] = -2 * self.Mpq(self.factor[i], self.S[i][0, 1]).imag
                kem[3, 2] = -(self.Mpq(self.factor[i], self.S[i][0, 0]).imag - self.Mpq(self.factor[i],
                                                                                        self.S[i][1, 1]).imag)
                kem[3, 3] = -(self.Mpq(self.factor[i], self.S[i][0, 0]).real + self.Mpq(self.factor[i],
                                                                                        self.S[i][1, 1]).real)

                self.__ke.append(kem)

        else:
            self.__ke = list()
            kem = np.zeros((4, 4))

            kem[0, 0] = -2 * self.Mpq(self.factor, self.S[0, 0]).real
            kem[0, 1] = 0
            kem[0, 2] = -self.Mpq(self.factor, self.S[0, 1]).real
            kem[0, 3] = -self.Mpq(self.factor, self.S[0, 0]).imag

            kem[1, 0] = 0
            kem[1, 1] = -2 * self.Mpq(self.factor, self.S[1, 1]).real
            kem[1, 2] = -self.Mpq(self.factor, self.S[1, 0]).real
            kem[1, 3] = -self.Mpq(self.factor, self.S[1, 0]).imag

            kem[2, 0] = -2 * self.Mpq(self.factor, self.S[1, 0]).real
            kem[2, 1] = -2 * self.Mpq(self.factor, self.S[0, 1]).real
            kem[2, 2] = -(self.Mpq(self.factor, self.S[0, 0]).real + self.Mpq(self.factor, self.S[1, 1]).real)
            kem[2, 3] = (self.Mpq(self.factor, self.S[0, 0]).imag - self.Mpq(self.factor, self.S[1, 1]).imag)

            kem[3, 0] = 2 * self.Mpq(self.factor, self.S[1, 0]).imag
            kem[3, 1] = -2 * self.Mpq(self.factor, self.S[0, 1]).imag
            kem[3, 2] = -(self.Mpq(self.factor, self.S[0, 0]).imag - self.Mpq(self.factor, self.S[1, 1]).imag)
            kem[3, 3] = -(self.Mpq(self.factor, self.S[0, 0]).real + self.Mpq(self.factor, self.S[1, 1]).real)

            self.__ke.append(kem)

        return self.__ke
