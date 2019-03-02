# -*- coding: utf-8 -*-
from __future__ import division

import sys
import warnings
from collections import namedtuple
from respy import Angles, Quantity, Conversion, units

import numpy as np
from pyrism.auxil import (SailResult, OPTICAL_RANGE,
                          Satellite, OpticalResult)
from respy.constants import deg_to_rad
from scipy.integrate import (quad)
from scipy.special import expi

from pyrism.volume.library import get_data_one, get_data_two

try:
    lib = get_data_two()
except IOError:
    lib = get_data_one()

# python 3.6 comparability
if sys.version_info < (3, 0):
    srange = xrange
else:
    srange = range


# ---- Scattering Coefficients ----
class VolScatt(Angles):
    """
    Compute volume scattering functions and interception coefficients
    for given solar zenith, viewing zenith, azimuth and leaf inclination angle (:cite:`Campbell.1986`,
    :cite:`Campbell.1990`, :cite:`Verhoef.1998`).

    Parameters
    ----------
    iza, vza, raa : int, float or ndarray
        Incidence (iza) and scattering (vza) zenith angle, as well as relative azimuth (raa) angle.
    angle_unit : {'DEG', 'RAD'}, optional
        * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
        * 'RAD': All input angles (iza, vza, raa) are in [RAD].

    Returns
    -------
    All returns are attributes!
    iza: ndarray
        Sun or incidence zenith angle in [RAD].
    vza : ndarray
        View or scattering zenith angle in [RAD].
    raa : ndarray
        Relative azimuth angle in [RAD].
    izaDeg : ndarray
        Sun or incidence zenith angle in [DEG].
    vzaDeg : ndarray
        View or scattering zenith angle in [DEG].
    raaDeg : ndarray
        Relative azimuth angle in [DEG].
    phi : ndarray
        Relative azimuth angle in a range between 0 and 2pi.
    chi_s : int, float or array_like
        Interception function  in the solar path.
    chi_o : int, float or array_like
        Interception function  in the view path.

    Note
    ----
    Hot spot direction is vza == iza and raa = 0.0

    See Also
    --------
    VolScatt.coef
    LIDF.campbell
    LIDF.verhoef
    LIDF.nilson

    """

    def __init__(self, iza, vza, raa, type='verhoef', n_elements=18, angle_unit='DEG', quantity=False, **kwargs):
        """
        Calculate the extinction and volume scattering coefficients (:cite:`Campbell.1986`,
        :cite:`Campbell.1990`, :cite:`Verhoef.1998`).

        Parameters
        ----------
        lidf_type : {'verhoef', 'campbell'}
            Define with which method the LIDF is calculated
        n_elements : int, optional
            Total number of equally spaced inclination angles. Default is 18.
        kwargs : dict
            Possible **kwargs from campbell method:
                * a : Mean leaf angle (degrees) use 57 for a spherical LIDF.

            Possible **kwargs from verhoef method:
                * a : Parameter a controls the average leaf inclination.
                * b : Parameter b influences the shape of the distribution (bimodality), but has no effect on the
                      average leaf inclination.

        Returns
        -------
        All returns are attributes!
        self.ks : int, float or array_like
            Volume scattering coefficient in incidence path.
        self.ko : int, float or array_like
            Volume scattering coefficient in scattering path.
        self.Fs : int, float or array_like
            Function to be multiplied by leaf reflectance to obtain the volume scattering.
        self.Ft : int, float or array_like
            Function to be multiplied by leaf transmittance to obtain the volume scattering.
        self.Fst : int, float or array_like
            Sum of Fs and Ft.

        See Also
        --------
        LIDF.campbell
        LIDF.verhoef
        LIDF.nilson

        """
        super(VolScatt, self).__init__(iza=iza, vza=vza, raa=raa, normalize=False, nbar=0.0, angle_unit=angle_unit,
                                       align=True)

        a = kwargs.pop('a', None)
        b = kwargs.pop('b', None)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if type == 'verhoef':
            if a is None or b is None:
                raise ValueError("for the verhoef function the parameter a and b must defined.")
            else:
                self.lidf = LIDF.verhoef(a, b, n_elements)

        elif type == 'campbell':
            if a is None:
                raise ValueError("for the campbell function the parameter alpha must defined.")
            else:
                self.lidf = LIDF.campbell(a, n_elements)

        else:
            raise AttributeError("lad_method must be verhoef, nilson or campbell")

        self.__output_as_quantity = quantity

        self.__kei = None
        self.__kev = None
        self.__bf = None
        self.__Fs = None
        self.__Ft = None
        self.__Fst = None

    # --------------------------------------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------------------------------------
    @property
    def kei(self):
        if self.__kei is None:
            self.__kei, self.__kev, self.__bf, self.__Fs, self.__Ft, self.__Fst = self.__coef()

        return self.__kei

    @property
    def kev(self):
        if self.__kei is None:
            self.__kei, self.__kev, self.__bf, self.__Fs, self.__Ft, self.__Fst = self.__coef()

        return self.__kev

    @property
    def bf(self):
        if self.__kei is None:
            self.__kei, self.__kev, self.__bf, self.__Fs, self.__Ft, self.__Fst = self.__coef()

        return self.__bf

    @property
    def Fs(self):
        if self.__kei is None:
            self.__kei, self.__kev, self.__bf, self.__Fs, self.__Ft, self.__Fst = self.__coef()

        return self.__Fs

    @property
    def Ft(self):
        if self.__kei is None:
            self.__kei, self.__kev, self.__bf, self.__Fs, self.__Ft, self.__Fst = self.__coef()

        return self.__Ft

    @property
    def Fst(self):
        if self.__kei is None:
            self.__kei, self.__kev, self.__bf, self.__Fs, self.__Ft, self.__Fst = self.__coef()

        return self.__Fst

    # --------------------------------------------------------------------------------------------------------
    # Callable Methods
    # --------------------------------------------------------------------------------------------------------
    def compute_extinction(self, ks, kt):
        if len(ks) != 2101:
            raise AssertionError(
                "ks must contain continuous leaf reflectance values from from 400 until 2500 nm with a length of "
                "2101. The actual length of ks is {0}".format(str(len(ks))))

        elif len(kt) != 2101:
            raise AssertionError(
                "kt must contain continuous leaf transmittance values from from 400 until 2500 nm with a length of "
                "2101. The actual length of kt is {0}".format(str(len(kt))))
        else:
            pass

        ddb = 0.5 * (1.0 + self.bf)
        ddf = 0.5 * (1.0 - self.bf)

        sigb = ddb * ks + ddf * kt
        sigf = ddf * ks + ddb * kt

        try:
            sigf[sigf == 0.0] = 1.e-36
            sigb[sigb == 0.0] = 1.0e-36
        except TypeError:
            sigf = max(1e-36, sigf)
            sigb = max(1e-36, sigb)

        att = 1. - sigf

        ke = np.sqrt(att ** 2. - sigb ** 2.)

        return np.asarray(ke).flatten()

    def volume(self, lza):
        """
        Compute volume scattering functions and interception coefficients
        for given solar zenith, viewing zenith, azimuth and leaf inclination angle (:cite:`Verhoef.1998`,
        :cite:`Campbell.1990`).

        Returns
        -------
        All returns are attributes!
        chi_s : float
            Interception function  in the solar path.
        chi_o : float
            Interception function  in the view path.
        frho : float
            Function to be multiplied by leaf reflectance to obtain the volume scattering.
        ftau : float
            Function to be multiplied by leaf transmittance to obtain the volume scattering.

        """
        cts = np.cos(self.iza.value)
        cto = np.cos(self.vza.value)
        sts = np.sin(self.iza.value)
        sto = np.sin(self.vza.value)
        cospsi = np.cos(self.raa.value)
        psir = self.raa.value
        clza = np.cos(np.radians(lza))
        slza = np.sin(np.radians(lza))
        cs = clza * cts
        co = clza * cto
        ss = slza * sts
        so = slza * sto
        cosbts = np.zeros_like(ss) + 5.
        cosbto = np.zeros_like(so) + 5.
        bts = np.zeros_like(cosbts)
        ds = np.zeros_like(ss)
        bto = np.zeros_like(ss)
        do_ = np.zeros_like(ss)

        for i in range(len(ss)):
            if np.abs(ss[i]) > 1e-6:
                cosbts[i] = -cs[i] / ss[i]

                if np.abs(cosbts[i]) < 1.0:
                    bts[i] = np.arccos(cosbts[i])
                    ds[i] = ss[i]

                else:
                    bts[i] = np.pi
                    ds[i] = cs[i]

        if i in range(len(so)):
            if np.abs(so[i]) > 1e-6:
                cosbto[i] = -co[i] / so[i]

                if np.abs(cosbto[i]) < 1:
                    bto[i] = np.arccos(cosbto[i])
                    do_[i] = so[i]

                else:
                    if self.vza[i] < (deg_to_rad * 90.0):
                        bto[i] = np.pi
                        do_[i] = co[i]
                    else:
                        bto[i] = 0
                        do_[i] = -co[i]

        chi_s = 2. / np.pi * ((bts - np.pi * 0.5) * cs + np.sin(bts) * ss)

        chi_o = 2.0 / np.pi * ((bto - np.pi * 0.5) * co + np.sin(bto) * so)
        btran1 = np.abs(bts - bto)
        btran2 = np.pi - np.abs(bts + bto - np.pi)
        bt1 = np.zeros_like(btran1)
        bt2 = np.zeros_like(btran1)
        bt3 = np.zeros_like(btran1)

        for i in range(len(btran1)):
            if psir[i] <= btran1[i]:
                bt1[i] = psir[i]
                bt2[i] = btran1[i]
                bt3[i] = btran2[i]
            else:
                bt1[i] = btran1[i]

                if psir[i] <= btran2[i]:
                    bt2[i] = psir[i]
                    bt3[i] = btran2[i]

                else:
                    bt2[i] = btran2[i]
                    bt3[i] = psir[i]

        t1 = 2. * cs * co + ss * so * cospsi
        # t2 = 0.
        t2 = np.zeros_like(bt2)
        for i in range(len(bt2)):
            if bt2[i] > 0.:
                t2[i] = np.sin(bt2[i]) * (2. * ds[i] * do_[i] + ss[i] * so[i] * np.cos(bt1[i]) * np.cos(bt3[i]))

        # if bt2 > 0.:
        #     t2 = np.sin(bt2) * (2. * ds * do_ + ss * so * np.cos(bt1) * np.cos(bt3))

        denom = 2. * np.pi ** 2
        frho = ((np.pi - bt2) * t1 + t2) / denom
        ftau = (-bt2 * t1 + t2) / denom
        frho[frho < 0.] = 0
        ftau[ftau < 0.] = 0

        return chi_s, chi_o, frho, ftau

    # --------------------------------------------------------------------------------------------------------
    # Private Methods
    # --------------------------------------------------------------------------------------------------------
    def __coef(self):

        kei = 0.
        kev = 0.
        bf = 0.
        Fs = 0.
        Ft = 0.

        n_angles = len(self.lidf)
        angle_step = float(90.0 / n_angles)
        litab = np.arange(n_angles) * angle_step + (angle_step * 0.5)

        for i, ili in enumerate(litab):
            ttl = 1. * ili
            cttl = np.cos(np.radians(ttl))
            # SAIL volume scattering phase function gives interception and portions to be multiplied by rho
            # and tau
            chi_s, chi_o, frho, ftau = self.volume(ttl)

            # Extinction coefficients
            ksli = chi_s / np.cos(self.iza.value)
            koli = chi_o / np.cos(self.vza.value)

            # Area scattering coefficient fractions
            sobli = frho * np.pi / (np.cos(self.iza.value) * np.cos(self.vza.value))
            sofli = ftau * np.pi / (np.cos(self.iza.value) * np.cos(self.vza.value))
            bfli = cttl ** 2.
            kei += ksli * float(self.lidf[i])
            kev += koli * float(self.lidf[i])
            bf += bfli * float(self.lidf[i])
            Fs += sobli * float(self.lidf[i])
            Ft += sofli * float(self.lidf[i])

            Fst = Fs + Ft

        return kei, kev, bf, Fs, Ft, Fst


# ---- LAD and LIDF Models ----
class LIDF:
    """
    Calculate several leaf area inclination density function  based on
    :cite:`Campbell.1990`, :cite:`Verhoef.1998` or :cite:`Nilson.1989`.

    See Also
    -------
    LIDF.campell
    LIDF.verhoef
    LIDF.nilson

    Note
    ----
    This class contains only static methods.

    """

    def __init__(self):
        pass

    @staticmethod
    def campbell(a, n_elements=18):
        """
        Calculate the Leaf Inclination Distribution Function based on the
        mean angle of ellipsoidal LIDF distribution.
        Parameters
        ----------
        a : float
            Mean leaf angle (degrees) use 57 for a spherical LIDF.
        n_elements : int
            Total number of equally spaced inclination angles .

        Returns
        -------
        lidf : np.ndarray
            Leaf Inclination Distribution Function for 18 equally spaced angles.

        """
        alpha = float(a)
        excent = np.exp(-1.6184e-5 * alpha ** 3. + 2.1145e-3 * alpha ** 2. - 1.2390e-1 * alpha + 3.2491)
        sum0 = 0.
        freq = np.zeros(n_elements)
        step = 90.0 / n_elements
        for i in srange(n_elements):
            tl1 = deg_to_rad * (i * step)
            tl2 = deg_to_rad * ((i + 1.) * step)
            x1 = excent / (np.sqrt(1. + excent ** 2. * np.tan(tl1) ** 2.))
            x2 = excent / (np.sqrt(1. + excent ** 2. * np.tan(tl2) ** 2.))
            if excent == 1.:
                freq[i] = abs(np.cos(tl1) - np.cos(tl2))
            else:
                alph = excent / np.sqrt(abs(1. - excent ** 2.))
                alph2 = alph ** 2.
                x12 = x1 ** 2.
                x22 = x2 ** 2.
                if excent > 1.:
                    alpx1 = np.sqrt(alph2 + x12)
                    alpx2 = np.sqrt(alph2 + x22)
                    dum = x1 * alpx1 + alph2 * np.log(x1 + alpx1)
                    freq[i] = abs(dum - (x2 * alpx2 + alph2 * np.log(x2 + alpx2)))
                else:
                    almx1 = np.sqrt(alph2 - x12)
                    almx2 = np.sqrt(alph2 - x22)
                    dum = x1 * almx1 + alph2 * np.arcsin(x1 / alph)
                    freq[i] = abs(dum - (x2 * almx2 + alph2 * np.arcsin(x2 / alph)))
        sum0 = np.sum(freq)
        lidf = np.zeros(n_elements)
        for i in srange(n_elements):
            lidf[i] = freq[i] / sum0

        return lidf

    @staticmethod
    def verhoef(a, b, n_elements=18):
        """
        Calculate the Leaf Inclination Distribution Function based on the
        Verhoef's bimodal LIDF distribution.

        Parameters
        ----------
        a, b : float
            Parameter a controls the average leaf inclination. Parameter b influences the shape of the distribution
            (bimodality), but has no effect on the average leaf inclination.
        n_elements : int
            Total number of equally spaced inclination angles.

        Returns
        -------
        LAD : list
            Leaf Inclination Distribution Function at equally spaced angles.

        Note
        ----
        The parameters must be chosen such that |a| + |b| < 1.
        Some possible distributions are [a, b]:
            * Planophile: [1, 0].
            * Erectophile: [-1, 0].
            * Plagiophile: [0,-1].
            * Extremophile: [0,1].
            * Spherical: [-0.35,-0.15].
            * Uniform: [0,0].

        """

        freq = 1.0
        step = 90.0 / n_elements
        lidf = np.zeros(n_elements) * 1.
        angles = (np.arange(n_elements) * step)[::-1]
        i = 0
        for angle in angles:
            tl1 = np.radians(angle)
            if a > 1.0:
                f = 1.0 - np.cos(tl1)
            else:
                eps = 1e-8
                delx = 1.0
                y = 1
                x = 2.0 * tl1
                p = float(x)
                while delx >= eps:
                    y = a * np.sin(x) + .5 * b * np.sin(2. * x)
                    dx = .5 * (y - x + p)
                    x = x + dx
                    delx = abs(dx)
                f = (2. * y + p) / np.pi
            freq = freq - f
            lidf[i] = freq
            freq = float(f)
            i += 1
        lidf = lidf[::-1]

        return lidf

    @staticmethod
    def nilson(lza, mla=None, eccentricity=0.5, scaling_factor=0.5, distribution='random'):
        """
        Leaf Angle Distributions (LAD) from Nilson and Kuusk.

        Note
        ----
        If mla is None, the default values are alculated by following distributions:
                * 'erectophile': 0
                * 'planophile': pi/2
                * 'plagiophile': pi/4
                * 'erectophile': 0
                * 'random' : This determines the output to 1
                * 'uniform' : This determines the output to 0.5

        Parameters
        ----------
        lza : int, float or ndarray
            Leaf zenith angle (lza).
        mla : int or float, optional
            Modal leaf angle in [Deg], Default is None (See Note).
        eccentricity : int or float (default = 0.5), optional
            Zero eccentricity is a spherical leaf angle distribution. An eccentricity
            of 1 is a 'needle'.
        scaling_factor : int or float (default = 0.5), optional
            Scaling factor (reflectance if lza = 0)
        distribution : {'erectophile', 'planophile', 'plagiophile', 'random', 'uniform'}, optional
            Default distribution which set the mla. Default is 'random'

        Returns
        -------
        LAD : int, float or array_like
            LAD integrated over a sphere (0 - pi/2)
        """

        if eccentricity > 1 or eccentricity < 0:
            raise AssertionError("eccentricity must between 0 and 1")

        if mla is None:
            if distribution == 'erectophile':
                if np.any(lza != np.pi / 2):
                    warnings.warn("Leaf normals should be mainly horizontal = 90°")
                mla = 0

            elif distribution == 'planophile':
                if lza != 0:
                    warnings.warn("Leaf normals should be mainly vertical = 0°")

                mla = np.pi / 2

            elif distribution == 'plagiophile':
                if lza != np.pi / 4:
                    warnings.warn("Leaf normals should be mainly at = 45°")

                mla = np.pi / 4

            else:
                raise ValueError("distribution must be erectophile, planophile, plagiophile, random or uniform")

        def __gfunc(lza, mla, e, b):
            return b / (1 - e ** 2 * np.sin((lza + mla) * np.pi / 180.0)) ** (1 / 2)

        def __integrant(x, lza, mla, e, b):
            return __gfunc(lza, mla, e, b) * np.cos(x) * np.sin(x)

        if distribution == 'random':
            return 1
        elif distribution == 'uniform':
            return 0.5
        else:
            if not isinstance(lza, np.ndarray):
                return quad(__integrant, 0, np.pi / 2, args=(lza, mla, eccentricity, scaling_factor))[0]
            else:
                lad_list = []
                for item in lza:
                    lad_list.append(quad(__integrant, 0, np.pi / 2, args=(item, mla, eccentricity, scaling_factor))[0])

                return np.asarray(lad_list)


class SAIL(Angles):
    """
    Run the SAIL radiative transfer model (See Note) (:cite:`GomezDans.2018`).

    Parameters
    ----------
    iza, vza, raa : int, float or ndarray
        Incidence (iza) and scattering (vza) zenith angle, as well as relative azimuth (raa) angle.
    ks, kt : array_like
        Continuous leaf reflection (ks) and leaf transmission (kt) values from from 400 until 2500 nm. One can use the
        output from PROSPECT class instance.
    lai : float
        Leaf area index.
    hotspot : float
        The hotspot parameter.
    rho_surface : array_like
        Continuous surface reflectance values from from 400 until 2500 nm. One can use the
        output from LSM class instance.
    lidf_type : {'verhoef', 'campbell'}, optional
        Define with which method the LIDF is calculated. Default is 'campbell'
    a, b : float, optional
        Parameter a and b depends on which lidf_type is applied:
            * If lidf_type is 'verhoef': Parameter a controls the average leaf inclination. Parameter b influences
              the shape of the distribution (bimodality), but has no effect on the average leaf inclination.
              The default values are for a uniform leaf distribution a = 0, b = 0.
            * If lidf_type is 'campbell': Parameter a is the mean leaf angle (degrees) use 57 for a spherical LIDF.
              The default value represents a spherical leaf distribution a = 57.

    skyl : float
        Fraction of diffuse shortwave radiation. Default is 0.2.
    normalize : boolean, optional
        Set to 'True' to make kernels 0 at nadir view illumination. Since all implemented kernels are normalized
        the default value is False.
    nbar : float, optional
        The sun or incidence zenith angle at which the isotropic term is set
        to if normalize is True. The default value is 0.0.
    angle_unit : {'DEG', 'RAD'}, optional
        * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
        * 'RAD': All input angles (iza, vza, raa) are in [RAD].

    Returns
    -------
    For more attributes see also pyrism.core.Kernel and pyrism.core.SailResult.

    See Also
    --------
    pyrism.core.Kernel
    pyrism.core.SailResult

    Note
    ----
    If the input parameter for ks and kt are the output from the class PROSPECT, SAIL will calculate the
    PROSAIL model.

    """

    def __init__(self, iza, vza, raa, ks, kt, lai, hotspot, rho_surface,
                 lidf_type='campbell', a=57, b=0, skyl=0.2, normalize=False, nbar=0.0, angle_unit='DEG'):

        super(SAIL, self).__init__(iza=iza, vza=vza, raa=raa, normalize=normalize, nbar=nbar, angle_unit=angle_unit,
                                   align=True)

        if len(ks) != 2101:
            raise AssertionError(
                "ks must contain continuous leaf reflectance values from from 400 until 2500 nm with a length of "
                "2101. The actual length of ks is {0}".format(str(len(ks))))

        elif len(kt) != 2101:
            raise AssertionError(
                "kt must contain continuous leaf transmittance values from from 400 until 2500 nm with a length of "
                "2101. The actual length of kt is {0}".format(str(len(kt))))

        elif len(rho_surface) != 2101:
            raise AssertionError(
                "rho_surface must contain continuous surface reflectance values from from 400 until 2500 nm with a "
                "length of 2101. The actual length of rho_surface is {0}".format(str(len(rho_surface))))

        else:
            pass

        self.ks = ks.value if hasattr(ks, 'quantity') else ks
        self.kt = kt.value if hasattr(kt, 'quantity') else kt

        self.lai = lai.value if hasattr(lai, 'quantity') else lai

        self.hotspot = hotspot.value if hasattr(hotspot, 'quantity') else hotspot

        self.rho_surface = rho_surface.value if hasattr(rho_surface, 'quantity') else rho_surface

        kwargs = {'a': a, 'b': b}

        self.VollScat = VolScatt(iza=iza, vza=vza, raa=raa, type=lidf_type, angle_unit=angle_unit, **kwargs)

        (tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo, rso, rsos, rsod, rddt, rsdt, rdot, rsodt,
         rsost, rsot, gammasdf, gammasdb, gammaso) = self.__calc()

        self.kt = tsstoo
        self.kt_iza = tss
        self.kt_vza = too

        # self.canopy = SailResult(BHR=rdd, BHT=tdd, DHR=rsd, DHT=tsd, HDR=rdo, HDT=tdo, BRF=rso)

        self.wavelength = OPTICAL_RANGE.wavelength
        self.frequency = OPTICAL_RANGE.frequency
        self.wavenumber = OPTICAL_RANGE.wavenumber

        canopy_ref = rdot * skyl + rsot * (1 - skyl)
        conversion = Conversion(canopy_ref, vza=self.vza, value_unit='BRF', angle_unit=angle_unit)

        self.__BRF = conversion.BRF

        self.__I = conversion.I
        self.__BSC = conversion.BSC
        self.__store()
        # self.BHR = rddt
        # self.DHR = rsdt
        # self.HDR = rdot

    @property
    def I(self):
        return self.__I

    @property
    def BRF(self):
        return self.__BRF

    @property
    def BSC(self):
        return self.__BSC

    @property
    def L8(self):
        return self.__L8

    @property
    def ASTER(self):
        return self.__ASTER

    def __calc(self):
        sdb = 0.5 * (self.VollScat.kei + self.VollScat.bf)
        sdf = 0.5 * (self.VollScat.kei - self.VollScat.bf)
        dob = 0.5 * (self.VollScat.kev + self.VollScat.bf)
        dof = 0.5 * (self.VollScat.kev - self.VollScat.bf)
        ddb = 0.5 * (1.0 + self.VollScat.bf)
        ddf = 0.5 * (1.0 - self.VollScat.bf)

        sigb = ddb * self.ks + ddf * self.kt
        sigf = ddf * self.ks + ddb * self.kt

        try:
            sigf[sigf == 0.0] = 1.e-36
            sigb[sigb == 0.0] = 1.0e-36
        except TypeError:
            sigf = max(1e-36, sigf)
            sigb = max(1e-36, sigb)

        att = 1. - sigf
        m = np.sqrt(att ** 2. - sigb ** 2.)

        self.ke = self.VollScat.kei

        sb = sdb * self.ks + sdf * self.kt
        sf = sdf * self.ks + sdb * self.kt
        vb = dob * self.ks + dof * self.kt
        vf = dof * self.ks + dob * self.kt
        w = self.VollScat.Fs * self.ks + self.VollScat.Ft * self.kt

        if np.all(self.lai <= 0):
            # No canopy...
            tss = 1
            too = 1
            tsstoo = 1
            rdd = 0
            tdd = 1
            rsd = 0
            tsd = 0
            rdo = 0
            tdo = 0
            rso = 0
            rsos = 0
            rsod = 0
            rddt = self.rho_surface
            rsdt = self.rho_surface
            rdot = self.rho_surface
            rsodt = 0
            rsost = self.rho_surface
            rsot = self.rho_surface
            gammasdf = 0
            gammaso = 0
            gammasdb = 0

            return [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo,
                    rso, rsos, rsod, rddt, rsdt, rdot, rsodt, rsost, rsot, gammasdf, gammasdb, gammaso]

        else:
            e1 = np.exp(-m * self.lai)
            e2 = e1 ** 2.
            rinf = (att - m) / sigb
            rinf2 = rinf ** 2.
            re = rinf * e1
            denom = 1. - rinf2 * e2

            J1ks = self.__Jfunc1(self.VollScat.kei, m, self.lai)
            J2ks = self.__Jfunc2(self.VollScat.kei, m, self.lai)
            J1ko = self.__Jfunc1(self.VollScat.kev, m, self.lai)
            J2ko = self.__Jfunc2(self.VollScat.kev, m, self.lai)

            Pss = (sf + sb * rinf) * J1ks
            Qss = (sf * rinf + sb) * J2ks
            Pv = (vf + vb * rinf) * J1ko
            Qv = (vf * rinf + vb) * J2ko

            tdd = (1. - rinf2) * e1 / denom
            rdd = rinf * (1. - e2) / denom
            tsd = (Pss - re * Qss) / denom
            rsd = (Qss - re * Pss) / denom
            tdo = (Pv - re * Qv) / denom
            rdo = (Qv - re * Pv) / denom

            gammasdf = (1. + rinf) * (J1ks - re * J2ks) / denom
            gammasdb = (1. + rinf) * (-re * J1ks + J2ks) / denom

            tss = np.exp(-self.VollScat.kei * self.lai)
            too = np.exp(-self.VollScat.kev * self.lai)
            z = self.__Jfunc2(self.VollScat.kei, self.VollScat.kev, self.lai)

            g1 = (z - J1ks * too) / (self.VollScat.kev + m)
            g2 = (z - J1ko * tss) / (self.VollScat.kei + m)

            Tv1 = (vf * rinf + vb) * g1
            Tv2 = (vf + vb * rinf) * g2
            T1 = Tv1 * (sf + sb * rinf)
            T2 = Tv2 * (sf * rinf + sb)
            T3 = (rdo * Qss + tdo * Pss) * rinf

            # Multiple scattering contribution to bidirectional canopy reflectance
            rsod = (T1 + T2 - T3) / (1. - rinf2)

            # Thermal "sod" quantity
            T4 = Tv1 * (1. + rinf)
            T5 = Tv2 * (1. + rinf)
            T6 = (rdo * J2ks + tdo * J1ks) * (1. + rinf) * rinf
            gammasod = (T4 + T5 - T6) / (1. - rinf2)

            # Treatment of the hotspot-effect
            alf = 1e36

            # Apply correction 2/(K+k) suggested by F.-M. Breon
            cts, cto, ctscto, tants, tanto, cospsi, dso = self.__define_geometric_constants(self.izaDeg.value,
                                                                                            self.vzaDeg.value,
                                                                                            self.raaDeg.value)

            if self.hotspot > 0.:
                alf = (dso / self.hotspot) * 2. / (self.VollScat.kei + self.VollScat.kev)

            if alf == 0.:
                # The pure hotspot
                tsstoo = tss
                sumint = (1. - tss) / (self.VollScat.kei * self.lai)
            else:
                # Outside the hotspot
                tsstoo, sumint = self.__hotspot_calculations(alf, self.lai, self.VollScat.kev,
                                                             self.VollScat.kei)

            # Bidirectional reflectance
            # Single scattering contribution
            rsos = w * self.lai * sumint
            gammasos = self.VollScat.kev * self.lai * sumint

            # Total canopy contribution
            rso = rsos + rsod
            gammaso = gammasos + gammasod

            # Interaction with the soil
            dn = 1. - self.rho_surface * rdd

            try:
                dn[dn < 1e-36] = 1e-36
            except TypeError:
                dn = max(1e-36, dn)

            rddt = rdd + tdd * self.rho_surface * tdd / dn
            rsdt = rsd + (tsd + tss) * self.rho_surface * tdd / dn
            rdot = rdo + tdd * self.rho_surface * (tdo + too) / dn
            rsodt = ((tss + tsd) * tdo + (
                    tsd + tss * self.rho_surface * rdd) * too) * self.rho_surface / dn
            rsost = rso + tsstoo * self.rho_surface
            rsot = rsost + rsodt

            return [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo,
                    rso, rsos, rsod, rddt, rsdt, rdot, rsodt, rsost, rsot, gammasdf, gammasdb, gammaso]

    def __define_geometric_constants(self, tts, tto, psi):
        cts = np.cos(np.radians(tts))
        cto = np.cos(np.radians(tto))
        ctscto = cts * cto
        tants = np.tan(np.radians(tts))
        tanto = np.tan(np.radians(tto))
        cospsi = np.cos(np.radians(psi))
        dso = np.sqrt(tants ** 2. + tanto ** 2. - 2. * tants * tanto * cospsi)
        return cts, cto, ctscto, tants, tanto, cospsi, dso

    def __hotspot_calculations(self, alf, lai, ko, ks):
        fhot = lai * np.sqrt(ko * ks)
        # Integrate by exponential Simpson method in 20 steps the steps are arranged according to equal
        # partitioning of the slope of the joint probability function

        x1 = 0.
        y1 = 0.
        f1 = 1.
        fint = (1. - np.exp(-alf)) * .05
        sumint = 0.
        for istep in srange(1, 21):
            if istep < 20:
                x2 = -np.log(1. - istep * fint) / alf
            else:
                x2 = 1.
            y2 = -(ko + ks) * lai * x2 + fhot * (1. - np.exp(-alf * x2)) / alf
            f2 = np.exp(y2)
            sumint = sumint + (f2 - f1) * (x2 - x1) / (y2 - y1)
            x1 = x2
            y1 = y2
            f1 = f2
        tsstoo = f1
        if np.isnan(sumint):
            sumint = 0.
        return tsstoo, sumint

    def __Jfunc1(self, k, l, t):
        """J1 function with avoidance of singularity problem."""

        try:
            nb = len(l)
        except TypeError:
            nb = 1
        del_ = (k - l) * t
        if nb > 1:
            result = np.zeros(nb)
            result[np.abs(del_) > 1e-3] = (np.exp(-l[np.abs(del_) > 1e-3] * t) -
                                           np.exp(-k * t)) / (k - l[np.abs(del_) > 1e-3])
            result[np.abs(del_) <= 1e-3] = 0.5 * t * (np.exp(-k * t) +
                                                      np.exp(-l[np.abs(del_) <= 1e-3] * t)) * \
                                           (1. - (del_[np.abs(del_) <= 1e-3] ** 2.) / 12.)
        else:
            if np.abs(del_) > 1e-3:
                result = (np.exp(-l * t) - np.exp(-k * t)) / (k - l)
            else:
                result = 0.5 * t * (np.exp(-k * t) + np.exp(-l * t)) * (1. - (del_ ** 2.) / 12.)
        return result

    def __Jfunc2(self, k, l, t):
        """J2 function."""
        return (1. - np.exp(-(k + l) * t)) / (k + l)

    def __store(self):
        sat_I = Satellite(self.I, name='Intensity')
        sat_BRF = Satellite(self.BRF, name='Bidirectional Reflectance Factor')
        sat_BSC = Satellite(self.BSC, name='Backscattering Coefficient')

        self.__L8 = OpticalResult(I=sat_I.L8, BRF=sat_BRF.L8, BSC=sat_BSC.L8,
                                  ndvi=sat_I.ndvi(), sr=sat_I.sr())
        self.__ASTER = OpticalResult(I=sat_I.ASTER, BRF=sat_BRF.ASTER, BSC=sat_BSC.ASTER,
                                     ndvi=sat_I.ndvi(satellite="ASTER"), sr=sat_I.sr(satellite="ASTER"))


class PROSPECT:
    """
    PROSPECT D and 5 (including carotenoids and brown pigments) version b (october, 20th 2009) (:cite:`Jacquemoud.1990`,
    :cite:`Feret.2008`, :cite:`Baret.`)

    Parameters
    ----------
    N : int, float, object
        Leaf structure parameter.
    Cab : int, float, object
        Chlorophyll a+b content.
    Cxc : int, float, object
        Carotenoids content.
    Cbr : int, float, object
        Brown pigments content in arbitrary units.
    Cw : int, float, object
        Equivalent water thickness.
    Cm : int, float, object
        Dry matter content
    alpha : int, float, object
        Mean leaf angle (degrees) use 57 for a spherical LIDF. Default is 40.
    version : {'5', 'D'}
        PROSPECT version. Default is '5'.

    Returns
    -------
    All returns are attributes!
    L8.Bx.kx : namedtuple (with dot access)
        Landsat 8 average kx (ks, kt, ke) values for Bx band (B2 until B7):
    ASTER.Bx.kx : namedtuple (with dot access)
        ASTER average kx (ks, kt, ke) values for Bx band (B1 until B9):
    l : array_like
        Continuous Wavelength from 400 until 2500 nm.
    kt : array_like
        Continuous Transmission from 400 until 2500 nm.
    ks : array_like
        Continuous Scattering from 400 until 2500 nm.
    ke : array_like
        Continuous Extinction from 400 until 2500 nm.
    ka : array_like
        Continuous Absorption from 400 until 2500 nm.
    om : array_like
        Continuous Omega value in terms of Radar from 400 until 2500 nm.

    """

    def __init__(self, N, Cab, Cxc, Cbr, Cw, Cm, Can=0, alpha=40, version='5'):

        self.__N = N if not hasattr(N, 'quantity') else N.value
        self.__Cab = Cab if not hasattr(Cab, 'quantity') else Cab.value
        self.__Cxc = Cxc if not hasattr(Cxc, 'quantity') else Cxc.value
        self.__Cbr = Cbr if not hasattr(Cbr, 'quantity') else Cbr.value
        self.__Cw = Cw if not hasattr(Cw, 'quantity') else Cw.value
        self.__Cm = Cm if not hasattr(Cm, 'quantity') else Cm.value
        self.__Can = Can if not hasattr(Can, 'quantity') else Can.value
        self.__alpha = alpha if not hasattr(alpha, 'quantity') else alpha.value
        self.ver = version

        self.wavelength = OPTICAL_RANGE.wavelength
        self.frequency = OPTICAL_RANGE.frequency
        self.wavenumber = OPTICAL_RANGE.wavenumber

        if self.ver != '5' and self.ver != 'D':
            raise ValueError("version must be '5' for PROSPECT 5 or 'D' for PROSPECT D. "
                             "The actual version is: {}".format(str(self.ver)))
        self.__set_coef()
        self.__pre_process()
        self.__calc()
        self.__store()

    @property
    def N(self):
        return Quantity(self.__N, name="Leaf Structure Parameter")

    @property
    def Cab(self):
        return Quantity(self.__Cab, unit=units.ug / units.cm ** 2, name="Chlorophyll Content")

    @property
    def Cxc(self):
        return Quantity(self.__Cxc, unit=units.ug / units.cm ** 2, name="Carotenoid Content")

    @property
    def Cbr(self):
        return Quantity(self.__Cbr, name="Brown Pigments")

    @property
    def Cw(self):
        return Quantity(self.__Cw, unit="cm", name="Equivalent Water Thickness")

    @property
    def Cm(self):
        return Quantity(self.__Cm, unit=units.g / units.cm ** 2, name="Leaf Mass")

    @property
    def Can(self):
        return Quantity(self.__Can, unit=units.ug / units.cm ** 2, name="Anthocyanins")

    @property
    def alpha(self):
        return Quantity(self.__alpha, unit='deg', name="")

    def __set_coef(self):

        if self.ver == 'D' and self.__Can == 0:
            raise AssertionError("For PROSPECT version D is the Anthocyanins value mandatory (!=0)")

        if self.ver == '5':
            self.__KN = lib.p5.KN
            self.__Kab = lib.p5.Kab
            self.__Kxc = lib.p5.Kxc
            self.__Kbr = lib.p5.Kbr
            self.__Kw = lib.p5.Kw
            self.__Km = lib.p5.Km
            self.__Kan = np.zeros_like(self.__Km)

        if self.ver == 'D':
            self.__KN = lib.pd.KN
            self.__Kab = lib.pd.Kab
            self.__Kxc = lib.pd.Kxc
            self.__Kbr = lib.pd.Kbr
            self.__Kw = lib.pd.Kw
            self.__Km = lib.pd.Km
            self.__Kan = lib.pd.Kan

        self.n_elems_list = [len(spectrum) for spectrum in
                             [self.__KN, self.__Kab, self.__Kxc, self.__Kbr, self.__Kw, self.__Km, self.__Kan]]

    def __pre_process(self):
        kall = (self.__Cab * self.__Kab + self.__Cxc * self.__Kxc + self.__Can * self.__Kan + self.__Cbr * self.__Kbr
                + self.__Cw * self.__Kw + self.__Cm * self.__Km) / self.__N

        j = kall > 0
        t1 = (1 - kall) * np.exp(-kall)
        t2 = kall ** 2 * (-expi(-kall))
        tau = np.ones_like(t1)
        tau[j] = t1[j] + t2[j]

        self.__r, self.__t, self.__ra, self.__ta, self.__denom = self.__refl_trans_one_layer(self.__alpha,
                                                                                             self.__KN, tau)

    def __calctav(self, alpha, KN):
        """
        Note
        ----
        Stern F. (1964), Transmission of isotropic radiation across an
        interface between two dielectrics, Appl. Opt., 3(1):111-113.
        Allen W.A. (1973), Transmission of isotropic light across a
        dielectric surface in two and three dimensions, J. Opt. Soc. Am.,
        63(6):664-666.
        """
        # rd = pi/180 np.deg2rad
        n2 = KN * KN
        npx = n2 + 1
        nm = n2 - 1
        a = (KN + 1) * (KN + 1) / 2.
        k = -(n2 - 1) * (n2 - 1) / 4.
        sa = np.sin(np.deg2rad(alpha))

        if alpha != 90:
            b1 = np.sqrt((sa * sa - npx / 2) * (sa * sa - npx / 2) + k)
        else:
            b1 = 0.
        b2 = sa * sa - npx / 2
        b = b1 - b2
        b3 = b ** 3
        a3 = a ** 3
        ts = (k ** 2 / (6 * b3) + k / b - b / 2) - (k ** 2. / (6 * a3) + k / a - a / 2)

        tp1 = -2 * n2 * (b - a) / (npx ** 2)
        tp2 = -2 * n2 * npx * np.log(b / a) / (nm ** 2)
        tp3 = n2 * (1 / b - 1 / a) / 2
        tp4 = 16 * n2 ** 2 * (n2 ** 2 + 1) * np.log((2 * npx * b - nm ** 2) / (2 * npx * a - nm ** 2)) / (
                npx ** 3 * nm ** 2)
        tp5 = 16 * n2 ** 3 * (1. / (2 * npx * b - nm ** 2) - 1 / (2 * npx * a - nm ** 2)) / (npx ** 3)
        tp = tp1 + tp2 + tp3 + tp4 + tp5
        tav = (ts + tp) / (2 * sa ** 2)

        return tav

    def __refl_trans_one_layer(self, alpha, KN, tau):
        # <Help and Info Section> -----------------------------------------
        """
        Note
        ----
        Reflectance and transmittance of one layer.

        Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969),
        Interaction of isotropic ligth with a compact plant leaf, J. Opt.
        Soc. Am., 59(10):1376-1379.
        """
        talf = self.__calctav(self.__alpha, KN)
        ralf = 1.0 - talf
        t12 = self.__calctav(90, KN)
        r12 = 1. - t12
        t21 = t12 / (KN * KN)
        r21 = 1 - t21

        # top surface side
        denom = 1. - r21 * r21 * tau * tau
        Ta = talf * tau * t21 / denom
        Ra = ralf + r21 * tau * Ta

        # bottom surface side
        t = t12 * tau * t21 / denom
        r = r12 + r21 * tau * t

        return r, t, Ra, Ta, denom

    def __calc(self):
        # <Help and Info Section> -----------------------------------------
        """
        Note
        ----
        Reflectance and transmittance of N layers
        Stokes equations to compute properties of next N-1 layers (N real)
        Normal case

        Stokes G.G. (1862), On the intensity of the light reflected from
        or transmitted through a pile of plates, Proc. Roy. Soc. Lond.,
        11:545-556.
        """

        D = np.sqrt((1 + self.__r + self.__t) * (1 + self.__r - self.__t) * (1. - self.__r + self.__t) * (
                1. - self.__r - self.__t))
        rq = self.__r * self.__r
        tq = self.__t * self.__t
        a = (1 + rq - tq + D) / (2 * self.__r)
        b = (1 - rq + tq + D) / (2 * self.__t)

        bNm1 = np.power(b, self.__N - 1)
        bN2 = bNm1 * bNm1
        a2 = a * a
        denom = a2 * bN2 - 1
        Rsub = a * (bN2 - 1) / denom
        Tsub = bNm1 * (a2 - 1) / denom

        # Case of zero absorption
        j = self.__r + self.__t >= 1.
        Tsub[j] = self.__t[j] / (self.__t[j] + (1 - self.__t[j]) * (self.__N - 1))
        Rsub[j] = 1 - Tsub[j]

        # Reflectance and transmittance of the leaf: combine top layer with next N-1 layers
        denom = 1 - Rsub * self.__r

        self.__kt = self.__ta * Tsub / denom
        self.__ks = self.__ra + self.__ta * Rsub * self.__t / denom
        self.__ka = 1 - self.__ks - self.__kt
        self.__ke = self.__ks + self.__ka
        self.__omega = self.__ks / self.__ke

        array = np.asarray([self.__ks, self.__kt, self.__ka, self.__ke, self.__omega], dtype=np.double)
        self.array = array.transpose()

    @property
    def omega(self):
        return Quantity(self.__omega, name='Single Scattering Albedo')

    @property
    def ke(self):
        return Quantity(self.__ke, name='Attenuation Coefficient')

    @property
    def ks(self):
        return Quantity(self.__ks, name='Scattering Coefficient (Leaf Reflectance)')

    @property
    def ka(self):
        return Quantity(self.__ka, name='Absorption Coefficient')

    @property
    def kt(self):
        return Quantity(self.__kt, name='Transmission Coefficient (Transmittance)')

    @property
    def L8(self):
        return self.__L8

    @property
    def ASTER(self):
        return self.__ASTER

    def __store(self):

        sat_ke = Satellite(self.ke, name='Attenuation Coefficient')
        sat_ks = Satellite(self.ks, name='Scattering Coefficient (Leaf Reflectance)')
        sat_ka = Satellite(self.ka, name='Absorption Coefficient')
        sat_kt = Satellite(self.kt, name='Transmission Coefficient (Transmittance)')
        sat_omega = Satellite(self.omega, name='Single Scattering Albedo')

        self.__L8 = OpticalResult(ke=sat_ke.L8, ks=sat_ks.L8, ka=sat_ka.L8, kt=sat_kt.L8, omega=sat_omega.L8,
                                  ndvi=sat_ks.ndvi(), sr=sat_ks.sr())
        self.__ASTER = OpticalResult(ke=sat_ke.ASTER, ks=sat_ks.ASTER, ka=sat_ka.ASTER, kt=sat_kt.ASTER,
                                     omega=sat_omega.ASTER, ndvi=sat_ks.ndvi(satellite='ASTER'),
                                     sr=sat_ks.sr(satellite='ASTER'))
