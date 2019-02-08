"""
T-Matrix scattering from non-spherical particles.
"""

from __future__ import division

import sys
import warnings

import numpy as np
from pyrism.core.tma import (NMAX_VEC_WRAPPER, SZ_S_VEC_WRAPPER, SZ_AF_VEC_WRAPPER, DBLQUAD_Z_S_WRAPPER,
                             XSEC_QS_S_WRAPPER, XSEC_ASY_S_WRAPPER, XSEC_ASY_AF_WRAPPER, XSEC_QE_WRAPPER,
                             XSEC_QSI_WRAPPER, equal_volume_from_maximum_wrapper)
from respy import Angles, EM
from respy.constants import pi as PI
from respy.units import Quantity
from respy.util import align_all

from pyrism.scattering.tmat.orientation import Orientation
from pyrism.scattering.tmat.tm_auxiliary import param_radius_type, param_shape, param_orientation

# python 3.6 comparability
if sys.version_info < (3, 0):
    srange = xrange
else:
    srange = range

warnings.simplefilter('default')


class TMatrix(Angles, object):
    """T-Matrix scattering from non-spherical particles.
    Class for simulating scattering from non-spherical particles with the
    T-Matrix method. Uses a wrapper to the Fortran code by M. Mishchenko.

    Attributes
    ----------
    TMatrix.iza, TMatrix.vza, TMatrix.raa, TMatrix.iaa, TMatrix.vaa, TMatrix.alpha, TMatrix.beta: array_like
        Incidence (iza) and scattering (vza) zenith angle, relative azimuth (raa) angle, incidence and viewing
        azimuth angle (ira, vra) in [RAD].
    TMatrix.izaDeg, TMatrix.vzaDeg, TMatrix.raaDeg, TMatrix.iaaDeg, TMatrix.vaaDeg, TMatrix.alphaDeg, TMatrix.betaDeg: array_like
        SIncidence (iza) and scattering (vza) zenith angle, relative azimuth (raa) angle, incidence and viewing
        azimuth angle (ira, vra) in [DEG].
    TMatrix.phi : array_like
        Relative azimuth angle in a range between 0 and 2pi.
    TMatrix.B, TMatrix.BDeg : array_like
        The result of (1/cos(vza)+1/cos(iza)).
    TMatrix.mui, TMatrix.muv : array_like
        Cosine of iza and vza in [RAD].
    TMatrix.geometries : tuple
        If raa is defined it shows a tuple with (iza, vza, raa, alpha, beta) in [RAD]. If iaa and vaa is defined
        the tuple will be (iza, vza, iaa, vaa, alpha, beta) in [RAD]
    TMatrix.geometriesDeg : tuple
        If raa is defined it shows a tuple with (iza, vza, raa, alpha, beta) in [DEG]. If iaa and vaa is defined
        the tuple will be (iza, vza, iaa, vaa, alpha, beta) in [DEG]
    TMatrix.nbar : float
        The sun or incidence zenith angle at which the isotropic term is set
        to if normalize is True. You can change this attribute within the class.
    TMatrix.normalize : bool
        Set to 'True' to make kernels 0 at nadir view illumination. Since all implemented kernels are normalized
        the default value is False.
    TMatrix.dtype : numpy.dtype
        Desired data type of all values. This attribute is changeable.
    TMatrix.frequency : array_like
        Frequency. Access with respy.EMW.
    TMatrix.wavelength : array_like
        Wavelength. Access with respy.EMW.
    TMatrix.wavenumber : array_like
        Free space wavenumber in unit of wavelength_unit.
    TMatrix.frequency_unit : str
        Frequency unit. Access with respy.EMW.
    TMatrix.wavelength_unit : str
        Wavelength unit. This is the same as radius unit. Access with respy.EMW.
    TMatrix.len : int
        Length of elements.
    TMatrix.shape : tuple
        Shape of elements.
    TMatrix.chi : array_like
        Free space wave number times the radius.
    TMatrix.factor : array_like
        Pre-factor to calculate the extinction matrix: (2 * PI * N * 1j) / wavenumber
    TMatrix.nmax : array_like
        NMAX parameter.
    TMatrix.radius_unit : str
        Radius unit.
    TMatrix.radius : array_like
        Particle radius.
    TMatrix.radius_type : str
        Radius type.
    TMatrix.axis_ratio : array_like
        Axis ratio.
    TMatrix.shape_volume : str
        Shape of volume.
    TMatrix.eps : array_like
        The complex refractive index.
    TMatrix.orientation : str
        The function to use to compute the orientational scattering properties.
    TMatrix.orientation_pdf: callable
        Particle orientation Probability Density Function (PDF) for orientational averaging.
    TMatrix.n_alpha : int
        Number of integration points in the alpha Euler angle. Default is 5.
    TMatrix.n_beta : int
        Umber of integration points in the beta Euler angle. Default is 10.
    TMatrix.N : array_like
        Number of scatterer in unit volume
    TMatrix.verbose : bool
        Verbose.

    TMatrix.S : array_like
        Complex Scattering Matrix.
    TMatrix.Z : array_like
        Phase Matrix.
    TMatrix.SZ : list or array_like
         Complex Scattering Matrix and Phase Matrix.
    TMatrix.Snorm : array_like
        S at iza and vza == 0.
    TMatrix.Znorm : array_like
        Z at iza and vza == 0.
    TMatrix.dblquad : array_like
        Half space integrated Z.
    TMatrix.array : array_like
        Parameter Z as a 4xn array where the rows of the array are the row sums of the Z matrix.

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

    TMatrix.xs : list or array_like
        Scattering Cross Section in [cm^2] for VV and HH polarization.
    TMatrix.xe : list or array_like
        Extinction Cross Section in [cm^2] for VV and HH polarization.
    TMatrix.asy : list or array_like
        Asymetry Factor in [cm^2] for VV and HH polarization.
    TMatrix.I : list or array_like
        Scattering intensity for VV and HH polarization.

    Methods
    -------
    compute_SZ(izaDeg=None, vzaDeg=None, iaaDeg=None, vaaDeg=None, alphaDeg=None, betaDeg=None)
        Function to calculate SZ for different angles.

    See Also
    --------
    respy.Angles
    respy.EMW
    pyrism.Orientation

    """

    def __init__(self, iza, vza, iaa, vaa, frequency, radius, eps, alpha=0.0, beta=0.0, N=1,
                 radius_type='REV', shape='SPH', orientation='S', axis_ratio=1.0, orientation_pdf=None, n_alpha=5,
                 n_beta=10, angle_unit='DEG', frequency_unit='GHz', length_unit='m',
                 verbose=False):

        """T-Matrix scattering from non-spherical particles.

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
        N : int, array_like
            Number of scatterer in unit volume. Default is 1.
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
        angle_unit : {'DEG', 'RAD'}, optional
            * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
            * 'RAD': All input angles (iza, vza, raa) are in [RAD].
        frequency_unit : {'Hz', 'PHz', 'kHz', 'daHz', 'MHz', 'THz', 'hHz', 'GHz'}
            Unit of entered frequency. Default is 'GHz'.
        length_unit : {'dm', 'nm', 'cm', 'mm', 'm', 'km', 'um'}
            Unit of the radius in meter (m), centimeter (cm) or nanometer (nm).

        """
        # Check input parameter ------------------------------------------------------------------------------------
        if radius_type not in param_radius_type.keys():
            raise ValueError("Radius type must be {0}".format(param_radius_type.keys()))

        if shape not in param_shape.keys():
            raise ValueError("Shape must be {0}".format(param_shape.keys()))

        if orientation not in param_orientation:
            raise ValueError("Orientation must be {0}".format(param_orientation))

        # Define angles and align data -----------------------------------------------------------------------------
        input_data = (iza, vza, iaa, vaa, frequency, radius, axis_ratio, alpha, beta, eps.real, eps.imag, N)

        (iza, vza, iaa, vaa, frequency, radius, axis_ratio, alpha, beta, eps_real,
         eps_imag, N) = align_all(input_data, dtype=np.double)

        Angles.__init__(self, iza=iza, vza=vza, raa=None, iaa=iaa, vaa=vaa, alpha=alpha, beta=beta,
                        normalize=False, angle_unit=angle_unit, nbar=0.0)

        self.normalized_flag = False

        # Define Frequency -----------------------------------------------------------------------------------------
        self.__length_unit = length_unit

        self.EMW = EM(input=frequency, unit=frequency_unit, output=self.__length_unit)

        self.__frequency, self.__wavelength, self.__wavenumber = self.EMW.frequency, self.EMW.wavelength, self.EMW.wavenumber

        # Self Definitions -----------------------------------------------------------------------------------------
        self.__pol = 4

        self.__radius = Quantity(radius, unit=self.__length_unit)

        self.__verbose = verbose
        self.__radius_type = param_radius_type[radius_type]

        self.__axis_ratio = axis_ratio
        self.__shape_volume = param_shape[shape]

        self.ddelt = 1e-3
        self.ndgs = 2

        self.__orient = orientation
        self.__or_pdf = self.__get_pdf(orientation_pdf)
        self.__or_pdf_str = orientation_pdf

        self.__n_alpha = int(n_alpha)
        self.__n_beta = int(n_beta)

        self.__epsi = eps_imag
        self.__epsr = eps_real

        self.__N_unit = '1' + ' ' + '/' + ' ' + self.__length_unit + ' ' + '**' + ' ' + '3'

        self.__N = Quantity(N, unit=self.__N_unit)

        self.__factor = (2 * PI * self.__N * 1j) / self.__wavenumber
        self.__chi = self.__wavenumber * self.__radius

        # Calculations ---------------------------------------------------------------------------------------------
        self.__nmax = self.__NMAX()

        # None Definitions ------------------------------------------------------------------------------------------
        self.__S = None
        self.__Z = None
        self.__Snorm = None
        self.__Znorm = None
        self.__dblZi = None
        self.__xs = None
        self.__xas = None
        self.__xe = None
        self.__XI = None
        self.__ke = None
        self.__ks = None
        self.__ka = None
        self.__omega = None
        self.__kt = None

    # ------------------------------------------------------------------------------------------------------------------
    # Magic Methods
    # ------------------------------------------------------------------------------------------------------------------
    def __repr__(self):

        prefix = '<{0}>'.format(self.__class__.__name__)

        return prefix

    def __len__(self):
        return len(self.nmax)

    # ------------------------------------------------------------------------------------------------------------------
    # Property Calls
    # ------------------------------------------------------------------------------------------------------------------
    # Access to Array Specific Attributes ------------------------------------------------------------------------------
    @property
    def len(self):
        """
        Length of array

        Returns
        -------
        len : int
        """
        return self.nmax.shape[0]

    @property
    def shape(self):
        """
        Shape of array

        Returns
        -------
        shape : tuple
        """
        return self.S.shape

    # Auxiliary Properties- --------------------------------------------------------------------------------------------
    @property
    # @denormalized_decorator
    def chi(self):
        """
        Chi parameter: wavelength * radius

        Returns
        -------
        chi : respy.units.quantity.Quantity

        """
        self.__chi = self.wavenumber * self.radius

        return self.__chi

    @property
    # @denormalized_decorator
    def factor(self):
        """
        Factor to calculate the extinction matrix: (2*pi*N*1j)/wavenumber
        Returns
        -------

        """
        factor = (2 * PI * self.N) / self.wavenumber

        self.__factor = complex(0, factor)

        return self.__factor

    @property
    # @denormalized_decorator
    def nmax(self):
        return self.__nmax.base

    # Frequency Calls ----------------------------------------------------------------------------------------------
    @property
    # @denormalized_decorator
    def frequency(self):
        self.__frequency = self.EMW.frequency
        return self.__frequency

    @property
    # @denormalized_decorator
    def wavelength(self):
        self.__wavelength = self.EMW.wavelength
        return self.EMW.wavelength

    @property
    # @denormalized_decorator
    def wavenumber(self):
        self.__wavenumber = self.EMW.wavenumber
        return self.__wavenumber

    # Parameter Calls ----------------------------------------------------------------------------------------------
    @property
    # @denormalized_decorator
    def radius(self):
        return self.__radius

    @property
    def radius_type(self):
        return self.__radius_type

    @property
    # @denormalized_decorator
    def axis_ratio(self):
        return self.__axis_ratio

    @property
    def shape_volume(self):
        return self.__shape_volume

    @property
    # @denormalized_decorator
    def eps(self):
        eps = self.__epsr + self.__epsi * 1j
        return Quantity(eps, name='Relative Permittivity of Medium', constant=True)

    @property
    def orientation(self):
        return self.__orient

    @property
    def orientation_pdf(self):
        return self.__or_pdf

    @property
    def n_alpha(self):
        return self.__n_alpha

    @property
    def n_beta(self):
        return self.__n_beta

    @property
    # @denormalized_decorator
    def N(self):
        return self.__N

    @property
    def verbose(self):
        return self.__verbose

    @verbose.setter
    def verbose(self, value):
        if isinstance(value, bool):
            self.__verbose = value
        else:
            raise ValueError("The value must be type bool.")

    # Scattering and Phase Matrices ------------------------------------------------------------------------------------
    @property
    # @denormalized_decorator
    def S(self):
        """
        Scattering matrix.

        Returns
        -------
        S : respy.units.quantity.Quantity
        """
        if self.__S is None:
            self.__S, self.__Z = self.compute_SZ()

        return Quantity(self.__S, unit=self.EMW.wavelength.unit,
                        name="Scattering Matrix", constant=True)

    @property
    # @normalized_decorator
    def Z(self):
        """
        Phase matrix.

        Returns
        -------
        Z : array_like
        """
        if self.__Z is None:
            self.__S, self.__Z = self.compute_SZ()

        return Quantity(self.__Z, unit=self.EMW.wavelength.unit ** 2,
                        name="Phase Matrix", constant=True)

    @property
    def SZ(self):
        """
        Scattering and phase matrix.

        Returns
        -------
        S, Z : tuple
        """
        if self.__S is None:
            self.__S, self.__Z = self.compute_SZ()

        elif self.__Z is None:
            self.__S, self.__Z = self.compute_SZ()

        else:
            pass

        return (Quantity(self.__S, unit=self.EMW.wavelength.unit,
                         name="Scattering Matrix", constant=True),
                Quantity(self.__Z, unit=self.EMW.wavelength.unit ** 2,
                         name="Phase Matrix", constant=True))

    # Integration of S and Z -------------------------------------------------------------------------------------------
    @property
    # @normalized_decorator
    def dblquad(self):
        """
        Half space integration of the phase matrix in incidence direction.

        Returns
        -------
        dbl : list or array_like
        """
        if self.__dblZi is None:
            self.__dblZi = self.__dblquad()

        return Quantity(self.__dblZi, name="Half Space Integrated Phase Matrix",
                        constant=True)

    # Extinction and Scattering Matrices -------------------------------------------------------------------------------
    @property
    def ke(self):
        """
        Extinction matrix for the current setup, with polarization.

        Returns
        -------
        ke : MemoryView, double[:,:}
            MemoryView of type array([[VV, HH]])
        """
        if self.__ke is None:
            self.__ke = self.__N * self.xe

        self.__ke.set_name('Attenuation Coefficient (VV, HH)')
        self.__ke.set_constant(True)

        return self.__ke

    @property
    def ks(self):
        """
        Scattering matrix for the current setup, with polarization.

        Returns
        -------
        ks : MemoryView, double[:,:}
            MemoryView of type array([[VV, HH]])
        """
        if self.__ks is None:
            self.__ks = self.xs * self.__N

        self.__ks.set_name('Scattering Coefficient (VV, HH)')
        self.__ks.set_constant(True)

        return self.__ks

    @property
    def ka(self):
        """
        Absorption matrix for the current setup, with polarization.

        Returns
        -------
        ka : MemoryView, double[:,:}
            MemoryView of type array([[VV, HH]])
        """
        if self.__ka is None:
            self.__ka = self.ke - self.ks

        self.__ka.set_name('Absorption Coefficient (VV, HH)')
        self.__ka.set_constant(True)

        return self.__ka

    @property
    def omega(self):
        """
        Single scattering albedo matrix for the current setup, with polarization.

        Returns
        -------
        omega : MemoryView, double[:,:}
            MemoryView of type array([[VV, HH]])
        """
        if self.__omega is None:
            self.__omega = self.ks / self.ke

        self.__omega.set_name('Single Scattering Albedo (VV, HH)')
        self.__omega.set_constant(True)

        return self.__omega

    @property
    def kt(self):
        """
        Transmission matrix for the current setup, with polarization.

        Returns
        -------
        kt : MemoryView, double[:,:}
            MemoryView of type array([[VV, HH]])
        """
        if self.__kt is None:
            self.__kt = 1 - self.ke

        self.__kt.set_name('Transmission Coefficient (VV, HH)')
        self.__kt.set_constant(True)

        return self.__kt

    # Cross Section ----------------------------------------------------------------------------------------------------
    @property
    def xs(self):
        """
        Scattering cross section for the current setup, with polarization.

        Returns
        -------
        QS : array_like
        """

        if self.__xs is None:
            self.__xs = self.__QS()

        return Quantity(self.__xs, unit=self.EMW.wavelength.unit ** 2, name='Scattering Cross Section (VV, HH)',
                        constant=True)

    @property
    def asy(self):
        """
        Asymmetry parameter for the current setup, with polarization.

        Returns
        -------
        QAS : array_like
        """
        if self.__xs is None:
            self.__xs = self.__QS()

        if self.__xas is None:
            self.__xas = self.__QAS()

        return Quantity(self.__xas.base / self.__xs.base, unit=None, name='Asymmetry Parameter (VV, HH)',
                        constant=True)

    @property
    def xe(self):
        """
        Extinction cross section for the current setup, with polarization.

        Returns
        -------
        QE : array_like
        """

        if self.__xe is None:
            self.__xe = self.__QE()

        return Quantity(self.__xe, unit=self.EMW.wavelength.unit ** 2, name='Extinction Cross Section (VV, HH)',
                        constant=True)

    @property
    def xi(self):
        """
        Differential Scattering Cross Section.

        Returns
        -------
        I : array_like
        """

        if self.__XI is None:
            self.__XI = self.__I()

        return Quantity(self.__XI, unit=self.EMW.wavelength.unit ** 2,
                        name='Differential Scattering Cross Section (VV, HH)',
                        constant=True)

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

        Returns
        -------
        S, Z : array_like
            Three dimensional scattering (S) and phase (Z) matrix.

        Note
        ----
        If xzaDeg, xaaDeg, alpha or beta is None, the inputed angles in __init__ will be choose.

        !!! IMPORTANT !!!
        If the angles are NOT NONE, the new values will NOT be affect the property calls S, Z and SZ!

        """
        # if self.normalized_flag:
        #     self.normalize = True

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

        if self.__orient is 'S':
            S, Z = SZ_S_VEC_WRAPPER(self.__nmax, self.__wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)

        elif self.__orient is 'AF':
            S, Z = SZ_AF_VEC_WRAPPER(self.__nmax, self.__wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg,
                                     self.__n_alpha, self.__n_beta, self.__or_pdf)
        else:
            raise ValueError("Orientation must be S or AF.")

        # self.normalize = False

        return S, Z

    # ------------------------------------------------------------------------------------------------------------------
    #  Auxiliary functions and private methods
    # ------------------------------------------------------------------------------------------------------------------
    # NMAX, S and Z ----------------------------------------------------------------------------------------------------
    def __NMAX(self):
        """
        Calculate NMAX parameter.
        """
        if self.__radius_type == 2:
            # Maximum radius is not directly supported in the original
            # so we convert it to equal volume radius
            radius_type = 1
            radius = np.zeros_like(self.iza)

            for i, item in enumerate(self.__radius.value):
                radius[i] = equal_volume_from_maximum_wrapper(item, self.__axis_ratio[i], self.__shape_volume)

        else:
            radius_type = self.__radius_type
            radius = self.__radius.value

        # if self.normalized_flag:
        #     self.normalize = True

        nmax = NMAX_VEC_WRAPPER(radius=radius, radius_type=radius_type,
                                wavelength=self.__wavelength.value,
                                eps_real=self.__epsr, eps_imag=self.__epsi,
                                axis_ratio=self.__axis_ratio, shape=self.__shape_volume, verbose=self.verbose)

        self.__radius = Quantity(radius, unit=self.__length_unit)

        # self.normalize = False

        return nmax

    # Integration of Phase and Scattering Matrix -----------------------------------------------------------------------
    def __dblquad(self, iza_flag=True):

        if iza_flag:
            xzaDeg = self.vzaDeg
            xaaDeg = self.vaaDeg
        else:
            xzaDeg = self.izaDeg
            xaaDeg = self.iaaDeg

        if self.__orient is 'S':
            Z = DBLQUAD_Z_S_WRAPPER(self.__nmax, self.__wavelength.value, xzaDeg, xaaDeg, self.alphaDeg,
                                    self.betaDeg, iza_flag)
        elif self.__orient is 'AF':
            Z = DBLQUAD_Z_S_WRAPPER(self.__nmax, self.__wavelength.value, xzaDeg, xaaDeg, self.n_alpha, self.n_beta,
                                    self.__or_pdf,
                                    iza_flag)
        else:
            raise ValueError("Orientation must be S or AF.")

        return Z.base.reshape((self.len, 4, 4))

    # Cross Section ----------------------------------------------------------------------------------------------------
    def __QS(self):
        """
        Scattering cross section for the current setup, with polarization.
        """

        if self.__orient is 'S':
            QS = XSEC_QS_S_WRAPPER(self.nmax, self.wavelength, self.izaDeg, self.iaaDeg, self.alphaDeg, self.betaDeg,
                                   verbose=self.verbose)
        elif self.__orient is 'AF':
            QS = XSEC_QS_S_WRAPPER(self.nmax, self.wavelength, self.izaDeg, self.iaaDeg, self.n_alpha, self.n_beta,
                                   self.__or_pdf, verbose=self.verbose)
        else:
            raise ValueError("Orientation must be S or AF.")

        return QS

    def __QAS(self):
        """
        Asymetry cross section for the current setup, with polarization.
        """

        if self.__orient is 'S':
            QAS = XSEC_ASY_S_WRAPPER(self.nmax, self.wavelength, self.izaDeg, self.iaaDeg, self.alphaDeg, self.betaDeg,
                                     verbose=self.verbose)
        elif self.__orient is 'AF':
            QAS = XSEC_ASY_AF_WRAPPER(self.nmax, self.wavelength, self.izaDeg, self.iaaDeg, self.n_alpha, self.n_beta,
                                      self.__or_pdf, verbose=self.verbose)
        else:
            raise ValueError("Orientation must be S or AF.")

        return QAS

    def __QE(self):
        """
        Extinction cross section for the current setup, with polarization.
        """
        # if self.normalized_flag:
        #     izaDeg = np.append(self.izaDeg, 0)
        #     iaaDeg = np.append(self.iaaDeg, 0)
        # else:
        #     izaDeg = self.izaDeg
        #     iaaDeg = self.iaaDeg

        izaDeg = self.izaDeg
        iaaDeg = self.iaaDeg

        S, Z = self.compute_SZ(vzaDeg=izaDeg, vaaDeg=iaaDeg)

        if self.normalized_flag:
            S = S[0:-1]

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
