# -*- coding: utf-8 -*-
from __future__ import division

import sys
import warnings
from collections import namedtuple

import numpy as np
from scipy.integrate import (quad, dblquad)
from scipy.misc import factorial
from scipy.special import expi

from .library import get_data_one, get_data_two
from ..auxil import ReflectanceResult, EmissivityResult, SailResult
from radarpy import Angles, cot, rad, dB, BRF, BRDF

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

    def __init__(self, iza, vza, raa, angle_unit='DEG'):

        super(VolScatt, self).__init__(iza=iza, vza=vza, raa=raa, normalize=False, nbar=0.0, angle_unit=angle_unit,
                                       align=True)

    def coef(self, lidf_type='verhoef', n_elements=18, **kwargs):
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
        a = kwargs.pop('a', None)
        b = kwargs.pop('b', None)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if lidf_type == 'verhoef':
            if a is None or b is None:
                raise ValueError("for the verhoef function the parameter a and b must defined.")
            else:
                lidf = LIDF.verhoef(a, b, n_elements)

        elif lidf_type == 'campbell':
            if a is None:
                raise ValueError("for the campbell function the parameter alpha must defined.")
            else:
                lidf = LIDF.campbell(a, n_elements)

        else:
            raise AttributeError("lad_method must be verhoef, nilson or campbell")

        self.kei = 0.
        self.kev = 0.
        self.bf = 0.
        self.Fs = 0.
        self.Ft = 0.

        n_angles = len(lidf)
        angle_step = float(90.0 / n_angles)
        litab = np.arange(n_angles) * angle_step + (angle_step * 0.5)

        for i, ili in enumerate(litab):
            ttl = 1. * ili
            cttl = np.cos(np.radians(ttl))
            # SAIL volume scattering phase function gives interception and portions to be multiplied by rho
            # and tau
            self.chi_s, self.chi_o, self.frho, self.ftau = self.volume(ttl)

            # Extinction coefficients
            ksli = self.chi_s / np.cos(self.iza)
            koli = self.chi_o / np.cos(self.vza)

            # Area scattering coefficient fractions
            sobli = self.frho * np.pi / (np.cos(self.iza) * np.cos(self.vza))
            sofli = self.ftau * np.pi / (np.cos(self.iza) * np.cos(self.vza))
            bfli = cttl ** 2.
            self.kei += ksli * float(lidf[i])
            self.kev += koli * float(lidf[i])
            self.bf += bfli * float(lidf[i])
            self.Fs += sobli * float(lidf[i])
            self.Ft += sofli * float(lidf[i])

            self.Fst = self.Fs + self.Ft

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
        cts = np.cos(self.iza)
        cto = np.cos(self.vza)
        sts = np.sin(self.iza)
        sto = np.sin(self.vza)
        cospsi = np.cos(self.raa)
        psir = self.raa
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

        # if np.abs(ss) > 1e-6:
        #     cosbts = -cs / ss

        if i in range(len(so)):
            if np.abs(so[i]) > 1e-6:
                cosbto[i] = -co[i] / so[i]

                if np.abs(cosbto[i]) < 1:
                    bto[i] = np.arccos(cosbto[i])
                    do_[i] = so[i]

                else:
                    if self.vza[i] < rad(90.0):
                        bto[i] = np.pi
                        do_[i] = co[i]
                    else:
                        bto[i] = 0
                        do_[i] = -co[i]

        # cosbto = 5.
        # if np.abs(so) > 1e-6:
        #     cosbto = -co / so

        # if np.abs(cosbts) < 1.0:
        #     bts = np.arccos(cosbts)
        #     ds = ss
        #
        # else:
        #     bts = np.pi
        #     ds = cs

        chi_s = 2. / np.pi * ((bts - np.pi * 0.5) * cs + np.sin(bts) * ss)

        # if abs(cosbto) < 1.0:
        #     bto = np.arccos(cosbto)
        #     do_ = so
        #
        # else:
        #     if self.vza < rad(90.):
        #         bto = np.pi
        #         do_ = co
        #     else:
        #         bto = 0.0
        #         do_ = -co

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
        #
        # if psir <= btran1:
        #     bt1 = psir
        #     bt2 = btran1
        #     bt3 = btran2
        # else:
        #     bt1 = btran1
        #     if psir <= btran2:
        #         bt2 = psir
        #         bt3 = btran2
        #     else:
        #         bt2 = btran2
        #         bt3 = psir

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
        lidf : list
            Leaf Inclination Distribution Function for 18 equally spaced angles.

        """
        alpha = float(a)
        excent = np.exp(-1.6184e-5 * alpha ** 3. + 2.1145e-3 * alpha ** 2. - 1.2390e-1 * alpha + 3.2491)
        sum0 = 0.
        freq = np.zeros(n_elements)
        step = 90.0 / n_elements
        for i in srange(n_elements):
            tl1 = rad(i * step)
            tl2 = rad((i + 1.) * step)
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
    def nilson(self, lza, mla=None, eccentricity=0.5, scaling_factor=0.5, distribution='random'):
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
                "ks must contain continuous leaf reflectance values from from 400 until 2500 nm with a length of 2101. The actual length of ks is {0}".format(
                    str(len(ks))))

        elif len(kt) != 2101:
            raise AssertionError(
                "kt must contain continuous leaf transmitance values from from 400 until 2500 nm with a length of 2101. The actual length of kt is {0}".format(
                    str(len(kt))))

        elif len(rho_surface) != 2101:
            raise AssertionError(
                "rho_surface must contain continuous surface reflectance values from from 400 until 2500 nm with a length of 2101. The actual length of rho_surface is {0}".format(
                    str(len(rho_surface))))

        else:
            pass

        self.ks = ks
        self.kt = kt
        self.lai = lai
        self.hotspot = hotspot

        self.rho_surface = rho_surface
        self.VollScat = VolScatt(iza, vza, raa, angle_unit)

        if lidf_type is 'verhoef':
            self.VollScat.coef(a=a, b=b, lidf_type='verhoef')
        elif lidf_type is 'campbell':
            self.VollScat.coef(a=a, lidf_type='campbell')
        else:
            raise AssertionError("The lidf_type must be 'verhoef' or 'campbell'")

        tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo, rso, rsos, rsod, rddt, rsdt, rdot, rsodt, rsost, rsot, gammasdf, gammasdb, gammaso = self.__calc()

        self.kt = tsstoo
        self.kt_iza = tss
        self.kt_vza = too
        self.canopy = SailResult(BHR=rdd, BHT=tdd, DHR=rsd, DHT=tsd, HDR=rdo, HDT=tdo, BRF=rso)
        self.l = np.arange(400, 2501)

        canopy_ref = rdot * skyl + rsot * (1 - skyl)

        self.BRF = SailResult(ref=canopy_ref, refdB=dB(canopy_ref), L8=self.__store_L8(canopy_ref),
                              ASTER=self.__store_aster(canopy_ref))
        self.BRDF = SailResult(ref=canopy_ref / np.pi, refdB=dB(canopy_ref / np.pi),
                               L8=self.__store_L8(canopy_ref / np.pi),
                               ASTER=self.__store_aster(canopy_ref / np.pi))

        self.BHR = SailResult(ref=rddt, refdB=dB(rddt), L8=self.__store_L8(rddt), ASTER=self.__store_aster(rddt))
        self.DHR = SailResult(ref=rsdt, refdB=dB(rsdt), L8=self.__store_L8(rsdt), ASTER=self.__store_aster(rsdt))
        self.HDR = SailResult(ref=rdot, refdB=dB(rdot), L8=self.__store_L8(rdot), ASTER=self.__store_aster(rdot))

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
            cts, cto, ctscto, tants, tanto, cospsi, dso = self.__define_geometric_constants(self.izaDeg, self.vzaDeg,
                                                                                            self.raaDeg)

            if self.hotspot > 0.:
                alf = (dso / self.hotspot) * 2. / (self.VollScat.kei + self.VollScat.kev)

            if alf == 0.:
                # The pure hotspot
                tsstoo = tss
                sumint = (1. - tss) / (self.VollScat.kei * self.lai)
            else:
                # Outside the hotspot
                tsstoo, sumint = self.__hotspot_calculations(alf, self.lai, self.VollScat.kev, self.VollScat.kei)

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
            rsodt = ((tss + tsd) * tdo + (tsd + tss * self.rho_surface * rdd) * too) * self.rho_surface / dn
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
        # Integrate by exponential Simpson method in 20 steps the steps are arranged according to equal partitioning of the slope of the joint probability function
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

    def __store_aster(self, value):
        """
        Store the leaf reflectance for ASTER bands B1 - B9.
        """

        value = np.array([self.l, value])
        value = value.transpose()

        ASTER = namedtuple('ASTER', 'B1 B2 B3 B4 B5 B6 B7 B8 B9')

        b1 = (520, 600)
        b2 = (630, 690)
        b3 = (760, 860)
        b4 = (1600, 1700)
        b5 = (2145, 2185)
        b6 = (2185, 2225)
        b7 = (2235, 2285)
        b8 = (2295, 2365)
        b9 = (2360, 2430)

        ARefB1 = value[(value[:, 0] >= b1[0]) & (value[:, 0] <= b1[1])]
        ARefB2 = value[(value[:, 0] >= b2[0]) & (value[:, 0] <= b2[1])]
        ARefB3 = value[(value[:, 0] >= b3[0]) & (value[:, 0] <= b3[1])]
        ARefB4 = value[(value[:, 0] >= b4[0]) & (value[:, 0] <= b4[1])]
        ARefB5 = value[(value[:, 0] >= b5[0]) & (value[:, 0] <= b5[1])]
        ARefB6 = value[(value[:, 0] >= b6[0]) & (value[:, 0] <= b6[1])]
        ARefB7 = value[(value[:, 0] >= b7[0]) & (value[:, 0] <= b7[1])]
        ARefB8 = value[(value[:, 0] >= b8[0]) & (value[:, 0] <= b8[1])]
        ARefB9 = value[(value[:, 0] >= b9[0]) & (value[:, 0] <= b9[1])]

        return ASTER(ARefB1[:, 1].mean(), ARefB2[:, 1].mean(), ARefB3[:, 1].mean(), ARefB4[:, 1].mean(),
                     ARefB5[:, 1].mean(), ARefB6[:, 1].mean(), ARefB7[:, 1].mean(), ARefB8[:, 1].mean(),
                     ARefB9[:, 1].mean())

    def __store_L8(self, value):
        """
        Store the leaf reflectance for LANDSAT8 bands
        B2 - B7.
        """

        value = np.array([self.l, value])
        value = value.transpose()

        L8 = namedtuple('L8', 'B2 B3 B4 B5 B6 B7')

        b2 = (452, 452 + 60)
        b3 = (533, 533 + 57)
        b4 = (636, 636 + 37)
        b5 = (851, 851 + 28)
        b6 = (1566, 1566 + 85)
        b7 = (2107, 2107 + 187)

        LRefB2 = value[(value[:, 0] >= b2[0]) & (value[:, 0] <= b2[1])]
        LRefB3 = value[(value[:, 0] >= b3[0]) & (value[:, 0] <= b3[1])]
        LRefB4 = value[(value[:, 0] >= b4[0]) & (value[:, 0] <= b4[1])]
        LRefB5 = value[(value[:, 0] >= b5[0]) & (value[:, 0] <= b5[1])]
        LRefB6 = value[(value[:, 0] >= b6[0]) & (value[:, 0] <= b6[1])]
        LRefB7 = value[(value[:, 0] >= b7[0]) & (value[:, 0] <= b7[1])]

        return L8(LRefB2[:, 1].mean(), LRefB3[:, 1].mean(), LRefB4[:, 1].mean(), LRefB5[:, 1].mean(),
                  LRefB6[:, 1].mean(), LRefB7[:, 1].mean())


class PROSPECT:
    """
    PROSPECT D and 5 (including carotenoids and brown pigments) version b (october, 20th 2009) (:cite:`Jacquemoud.1990`,
    :cite:`Feret.2008`, :cite:`Baret.`)

    Parameters
    ----------
    N : int or float
        Leaf structure parameter.
    Cab : int or float
        Chlorophyll a+b content.
    Cxc : int or float
        Carotenoids content.
    Cbr : int or float
        Brown pigments content in arbitrary units.
    Cw : int or float
        Equivalent water thickness.
    Cm : int or float
        Dry matter content
    alpha : int
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

        self.N = N
        self.Cab = Cab
        self.Cxc = Cxc
        self.Cbr = Cbr
        self.Cw = Cw
        self.Cm = Cm
        self.Can = Can
        self.alpha = alpha
        self.ver = version

        self.l = np.arange(400, 2501)
        self.n_l = len(self.l)

        if self.ver != '5' and self.ver != 'D':
            raise ValueError("version must be '5' for PROSPECT 5 or 'D' for PROSPECT D. "
                             "The actual version is: {}".format(str(self.ver)))
        self.__set_coef()
        self.__pre_process()
        self.__calc()
        self.__store()

    def __set_coef(self):

        if self.ver == 'D' and self.Can == 0:
            raise AssertionError("For PROSPECT version D is the Anthocyanins value mandatory (!=0)")

        if self.ver == '5':
            self.KN = lib.p5.KN
            self.Kab = lib.p5.Kab
            self.Kxc = lib.p5.Kxc
            self.Kbr = lib.p5.Kbr
            self.Kw = lib.p5.Kw
            self.Km = lib.p5.Km
            self.Kan = np.zeros_like(self.Km)

        if self.ver == 'D':
            self.KN = lib.pd.KN
            self.Kab = lib.pd.Kab
            self.Kxc = lib.pd.Kxc
            self.Kbr = lib.pd.Kbr
            self.Kw = lib.pd.Kw
            self.Km = lib.pd.Km
            self.Kan = lib.pd.Kan

        self.n_elems_list = [len(spectrum) for spectrum in
                             [self.KN, self.Kab, self.Kxc, self.Kbr, self.Kw, self.Km, self.Kan]]

    def __pre_process(self):
        kall = (self.Cab * self.Kab + self.Cxc * self.Kxc + self.Can * self.Kan + self.Cbr * self.Kbr
                + self.Cw * self.Kw + self.Cm * self.Km) / self.N

        j = kall > 0
        t1 = (1 - kall) * np.exp(-kall)
        t2 = kall ** 2 * (-expi(-kall))
        tau = np.ones_like(t1)
        tau[j] = t1[j] + t2[j]

        self.r, self.t, self.Ra, self.Ta, self.denom = self.__refl_trans_one_layer(self.alpha, self.KN, tau)

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
        talf = self.__calctav(self.alpha, KN)
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

        D = np.sqrt((1 + self.r + self.t) * (1 + self.r - self.t) * (1. - self.r + self.t) * (1. - self.r - self.t))
        rq = self.r * self.r
        tq = self.t * self.t
        a = (1 + rq - tq + D) / (2 * self.r)
        b = (1 - rq + tq + D) / (2 * self.t)

        bNm1 = np.power(b, self.N - 1)
        bN2 = bNm1 * bNm1
        a2 = a * a
        denom = a2 * bN2 - 1
        Rsub = a * (bN2 - 1) / denom
        Tsub = bNm1 * (a2 - 1) / denom

        # Case of zero absorption
        j = self.r + self.t >= 1.
        Tsub[j] = self.t[j] / (self.t[j] + (1 - self.t[j]) * (self.N - 1))
        Rsub[j] = 1 - Tsub[j]

        # Reflectance and transmittance of the leaf: combine top layer with next N-1 layers
        denom = 1 - Rsub * self.r

        self.kt = self.Ta * Tsub / denom
        self.ks = self.Ra + self.Ta * Rsub * self.t / denom
        self.ka = 1 - self.ks - self.kt
        self.ke = self.ks + self.ka
        self.om = self.ks / self.ke

        self.int = [self.l, self.ks, self.kt, self.ka, self.ke, self.om]
        RT = np.asarray(self.int, dtype=np.float32)
        self.int = RT.transpose()

    def __store(self):
        """
        Store the leaf reflectance for ASTER bands B1 - B9 or LANDSAT8 bands
        B2 - B7.
        """

        ASTER = namedtuple('ASTER', 'B1 B2 B3 B4 B5 B6 B7 B8 B9')
        B1 = namedtuple('B1', 'ks kt ka ke omega')
        B2 = namedtuple('B2', 'ks kt ka ke omega')
        B3 = namedtuple('B3', 'ks kt ka ke omega')
        B4 = namedtuple('B4', 'ks kt ka ke omega')
        B5 = namedtuple('B5', 'ks kt ka ke omega')
        B6 = namedtuple('B6', 'ks kt ka ke omega')
        B7 = namedtuple('B7', 'ks kt ka ke omega')
        B8 = namedtuple('B8', 'ks kt ka ke omega')
        B9 = namedtuple('B9', 'ks kt ka ke omega')

        b1 = (520, 600)
        b2 = (630, 690)
        b3 = (760, 860)
        b4 = (1600, 1700)
        b5 = (2145, 2185)
        b6 = (2185, 2225)
        b7 = (2235, 2285)
        b8 = (2295, 2365)
        b9 = (2360, 2430)

        ARefB1 = self.int[(self.int[:, 0] >= b1[0]) & (self.int[:, 0] <= b1[1])]
        ARefB1 = [ARefB1[:, 1].mean(), ARefB1[:, 2].mean(), ARefB1[:, 3].mean(), ARefB1[:, 4].mean(),
                  ARefB1[:, 5].mean()]

        ARefB2 = self.int[(self.int[:, 0] >= b2[0]) & (self.int[:, 0] <= b2[1])]
        ARefB2 = [ARefB2[:, 1].mean(), ARefB2[:, 2].mean(), ARefB2[:, 3].mean(), ARefB2[:, 4].mean(),
                  ARefB2[:, 5].mean()]

        ARefB3 = self.int[(self.int[:, 0] >= b3[0]) & (self.int[:, 0] <= b3[1])]
        ARefB3 = [ARefB3[:, 1].mean(), ARefB3[:, 2].mean(), ARefB3[:, 3].mean(), ARefB3[:, 4].mean(),
                  ARefB3[:, 5].mean()]

        ARefB4 = self.int[(self.int[:, 0] >= b4[0]) & (self.int[:, 0] <= b4[1])]
        ARefB4 = [ARefB4[:, 1].mean(), ARefB4[:, 2].mean(), ARefB4[:, 3].mean(), ARefB4[:, 4].mean(),
                  ARefB4[:, 5].mean()]

        ARefB5 = self.int[(self.int[:, 0] >= b5[0]) & (self.int[:, 0] <= b5[1])]
        ARefB5 = [ARefB5[:, 1].mean(), ARefB5[:, 2].mean(), ARefB5[:, 3].mean(), ARefB5[:, 4].mean(),
                  ARefB5[:, 5].mean()]

        ARefB6 = self.int[(self.int[:, 0] >= b6[0]) & (self.int[:, 0] <= b6[1])]
        ARefB6 = [ARefB6[:, 1].mean(), ARefB6[:, 2].mean(), ARefB6[:, 3].mean(), ARefB6[:, 4].mean(),
                  ARefB6[:, 5].mean()]

        ARefB7 = self.int[(self.int[:, 0] >= b7[0]) & (self.int[:, 0] <= b7[1])]
        ARefB7 = [ARefB7[:, 1].mean(), ARefB7[:, 2].mean(), ARefB7[:, 3].mean(), ARefB7[:, 4].mean(),
                  ARefB7[:, 5].mean()]

        ARefB8 = self.int[(self.int[:, 0] >= b8[0]) & (self.int[:, 0] <= b8[1])]
        ARefB8 = [ARefB8[:, 1].mean(), ARefB8[:, 2].mean(), ARefB8[:, 3].mean(), ARefB8[:, 4].mean(),
                  ARefB8[:, 5].mean()]

        ARefB9 = self.int[(self.int[:, 0] >= b9[0]) & (self.int[:, 0] <= b9[1])]
        ARefB9 = [ARefB9[:, 1].mean(), ARefB9[:, 2].mean(), ARefB9[:, 3].mean(), ARefB9[:, 4].mean(),
                  ARefB9[:, 5].mean()]

        B1 = B1(ARefB1[0], ARefB1[1], ARefB1[2], ARefB1[3], ARefB1[4])
        B2 = B2(ARefB2[0], ARefB2[1], ARefB2[2], ARefB2[3], ARefB2[4])
        B3 = B3(ARefB3[0], ARefB3[1], ARefB3[2], ARefB3[3], ARefB3[4])
        B4 = B4(ARefB4[0], ARefB4[1], ARefB4[2], ARefB4[3], ARefB4[4])
        B5 = B5(ARefB5[0], ARefB5[1], ARefB5[2], ARefB5[3], ARefB5[4])
        B6 = B6(ARefB6[0], ARefB6[1], ARefB6[2], ARefB6[3], ARefB6[4])
        B7 = B7(ARefB7[0], ARefB7[1], ARefB7[2], ARefB7[3], ARefB7[4])
        B8 = B8(ARefB8[0], ARefB8[1], ARefB8[2], ARefB8[3], ARefB8[4])
        B9 = B9(ARefB9[0], ARefB9[1], ARefB9[2], ARefB9[3], ARefB9[4])

        self.ASTER = ASTER(B1, B2, B3, B4, B5, B6, B7, B8, B9)

        L8 = namedtuple('L8', 'B2 B3 B4 B5 B6 B7')
        B2 = namedtuple('B2', 'ks kt ka ke omega')
        B3 = namedtuple('B3', 'ks kt ka ke omega')
        B4 = namedtuple('B4', 'ks kt ka ke omega')
        B5 = namedtuple('B5', 'ks kt ka ke omega')
        B6 = namedtuple('B6', 'ks kt ka ke omega')
        B7 = namedtuple('B7', 'ks kt ka ke omega')

        b2 = (452, 452 + 60)
        b3 = (533, 533 + 57)
        b4 = (636, 636 + 37)
        b5 = (851, 851 + 28)
        b6 = (1566, 1566 + 85)
        b7 = (2107, 2107 + 187)

        LRefB2 = self.int[(self.int[:, 0] >= b2[0]) & (self.int[:, 0] <= b2[1])]
        LRefB2 = [LRefB2[:, 1].mean(), LRefB2[:, 2].mean(), LRefB2[:, 3].mean(), LRefB2[:, 4].mean(),
                  LRefB2[:, 5].mean()]

        LRefB3 = self.int[(self.int[:, 0] >= b3[0]) & (self.int[:, 0] <= b3[1])]
        LRefB3 = [LRefB3[:, 1].mean(), LRefB3[:, 2].mean(), LRefB3[:, 3].mean(), LRefB3[:, 4].mean(),
                  LRefB3[:, 5].mean()]

        LRefB4 = self.int[(self.int[:, 0] >= b4[0]) & (self.int[:, 0] <= b4[1])]
        LRefB4 = [LRefB4[:, 1].mean(), LRefB4[:, 2].mean(), LRefB4[:, 3].mean(), LRefB4[:, 4].mean(),
                  LRefB4[:, 5].mean()]

        LRefB5 = self.int[(self.int[:, 0] >= b5[0]) & (self.int[:, 0] <= b5[1])]
        LRefB5 = [LRefB5[:, 1].mean(), LRefB5[:, 2].mean(), LRefB5[:, 3].mean(), LRefB5[:, 4].mean(),
                  LRefB5[:, 5].mean()]

        LRefB6 = self.int[(self.int[:, 0] >= b6[0]) & (self.int[:, 0] <= b6[1])]
        LRefB6 = [LRefB6[:, 1].mean(), LRefB6[:, 2].mean(), LRefB6[:, 3].mean(), LRefB6[:, 4].mean(),
                  LRefB6[:, 5].mean()]

        LRefB7 = self.int[(self.int[:, 0] >= b7[0]) & (self.int[:, 0] <= b7[1])]
        LRefB7 = [LRefB7[:, 1].mean(), LRefB7[:, 2].mean(), LRefB7[:, 3].mean(), LRefB7[:, 4].mean(),
                  LRefB7[:, 5].mean()]

        B2 = B2(LRefB2[0], LRefB2[1], LRefB2[2], LRefB2[3], LRefB2[4])
        B3 = B3(LRefB3[0], LRefB3[1], LRefB3[2], LRefB3[3], LRefB3[4])
        B4 = B4(LRefB4[0], LRefB4[1], LRefB4[2], LRefB4[3], LRefB4[4])
        B5 = B5(LRefB5[0], LRefB5[1], LRefB5[2], LRefB5[3], LRefB5[4])
        B6 = B6(LRefB6[0], LRefB6[1], LRefB6[2], LRefB6[3], LRefB6[4])
        B7 = B7(LRefB7[0], LRefB7[1], LRefB7[2], LRefB7[3], LRefB7[4])

        self.L8 = L8(B2, B3, B4, B5, B6, B7)

    def select(self, mins=None, maxs=None, function='mean'):
        """
        Returns the means of the coefficients in range between min and max.

        Parameters
        ----------
        mins : int
            Lower bound of the wavelength (400 - 2500)
        maxs : int
            Upper bound of the wavelength (400 - 2500)

        function : {'mean'}, optional
            Specify  how the bands are calculated.

        Returns
        -------
        Band : array_like
            Reflectance in the selected range.
        """
        if function == 'mean':
            ranges = self.int[(self.int[:, 0] >= mins) & (self.int[:, 0] <= maxs)]
            ks = ranges[:, 1].mean()
            kt = ranges[:, 2].mean()
            ka = ranges[:, 3].mean()
            ke = ranges[:, 4].mean()
            om = ranges[:, 5].mean()

            ranges = [ks, kt, ka, ke, om]
            return np.asarray(ranges, dtype=np.float32)

    def indices(self):
        self.ndvi = (self.select(851, 879)[0] - self.select(636, 673)[0]) / (
                self.select(851, 879)[0] + self.select(636, 673)[0])

        return self.ndvi

    def cleanup(self, name):
        """Do cleanup for an attribute"""
        try:
            delattr(self, name)
        except TypeError:
            for item in name:
                delattr(self, item)
