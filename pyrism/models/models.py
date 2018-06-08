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
from ..core import (Kernel, Scattering, ReflectanceResult, EmissivityResult, SailResult, cot, rad, dB, BRDF, BRF)

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
class VolScatt(Kernel):
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

        super(VolScatt, self).__init__(iza, vza, raa, normalize=False, nbar=0.0, angle_unit=angle_unit, align=True)

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
            Volume scattering coeffient in incidence path.
        self.ko : int, float or array_like
            Volume scattering coeffient in scattering path.
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

        self.ks = 0.
        self.ko = 0.
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
            self.ks += ksli * float(lidf[i])
            self.ko += koli * float(lidf[i])
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
        cosbts = 5.
        if np.abs(ss) > 1e-6:
            cosbts = -cs / ss
        cosbto = 5.
        if np.abs(so) > 1e-6:
            cosbto = -co / so
        if np.abs(cosbts) < 1.0:
            bts = np.arccos(cosbts)
            ds = ss
        else:
            bts = np.pi
            ds = cs
        chi_s = 2. / np.pi * ((bts - np.pi * 0.5) * cs + np.sin(bts) * ss)
        if abs(cosbto) < 1.0:
            bto = np.arccos(cosbto)
            do_ = so
        else:
            if self.vza < rad(90.):
                bto = np.pi
                do_ = co
            else:
                bto = 0.0
                do_ = -co
        chi_o = 2.0 / np.pi * ((bto - np.pi * 0.5) * co + np.sin(bto) * so)
        btran1 = np.abs(bts - bto)
        btran2 = np.pi - np.abs(bts + bto - np.pi)
        if psir <= btran1:
            bt1 = psir
            bt2 = btran1
            bt3 = btran2
        else:
            bt1 = btran1
            if psir <= btran2:
                bt2 = psir
                bt3 = btran2
            else:
                bt2 = btran2
                bt3 = psir
        t1 = 2. * cs * co + ss * so * cospsi
        t2 = 0.
        if bt2 > 0.:
            t2 = np.sin(bt2) * (2. * ds * do_ + ss * so * np.cos(bt1) * np.cos(bt3))
        denom = 2. * np.pi ** 2
        frho = ((np.pi - bt2) * t1 + t2) / denom
        ftau = (-bt2 * t1 + t2) / denom
        if frho < 0.:
            frho = 0.
        if ftau < 0.:
            ftau = 0.

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


class SAIL(Kernel):
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
                 lidf_type='campbell', a=57, b=0, normalize=False, nbar=0.0, angle_unit='DEG'):

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

        self.BRF = SailResult(ref=rsot, refdB=dB(rsot), L8=self.__store_L8(rsot), ASTER=self.__store_aster(rsot))
        self.BRDF = SailResult(ref=rsot / np.pi, refdB=dB(rsot / np.pi), L8=self.__store_L8(rsot / np.pi),
                               ASTER=self.__store_aster(rsot / np.pi))
        self.BHR = SailResult(ref=rddt, refdB=dB(rddt), L8=self.__store_L8(rddt), ASTER=self.__store_aster(rddt))
        self.DHR = SailResult(ref=rsdt, refdB=dB(rsdt), L8=self.__store_L8(rsdt), ASTER=self.__store_aster(rsdt))
        self.HDR = SailResult(ref=rdot, refdB=dB(rdot), L8=self.__store_L8(rdot), ASTER=self.__store_aster(rdot))

    def __calc(self):
        sdb = 0.5 * (self.VollScat.ks + self.VollScat.bf)
        sdf = 0.5 * (self.VollScat.ks - self.VollScat.bf)
        dob = 0.5 * (self.VollScat.ko + self.VollScat.bf)
        dof = 0.5 * (self.VollScat.ko - self.VollScat.bf)
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

            J1ks = self.__Jfunc1(self.VollScat.ks, m, self.lai)
            J2ks = self.__Jfunc2(self.VollScat.ks, m, self.lai)
            J1ko = self.__Jfunc1(self.VollScat.ko, m, self.lai)
            J2ko = self.__Jfunc2(self.VollScat.ko, m, self.lai)

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

            tss = np.exp(-self.VollScat.ks * self.lai)
            too = np.exp(-self.VollScat.ko * self.lai)
            z = self.__Jfunc2(self.VollScat.ks, self.VollScat.ko, self.lai)

            g1 = (z - J1ks * too) / (self.VollScat.ko + m)
            g2 = (z - J1ko * tss) / (self.VollScat.ks + m)

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
                alf = (dso / self.hotspot) * 2. / (self.VollScat.ks + self.VollScat.ko)

            if alf == 0.:
                # The pure hotspot
                tsstoo = tss
                sumint = (1. - tss) / (self.VollScat.ks * self.lai)
            else:
                # Outside the hotspot
                tsstoo, sumint = self.__hotspot_calculations(alf, self.lai, self.VollScat.ko, self.VollScat.ks)

            # Bidirectional reflectance
            # Single scattering contribution
            rsos = w * self.lai * sumint
            gammasos = self.VollScat.ko * self.lai * sumint

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


class Rayleigh(Scattering):
    """
    Calculate the extinction coefficients in terms of Rayleigh
    scattering (:cite:`Ulaby.2015` and :cite:`Ulaby.2015b`).

    Parameters
    ----------
    frequency : int or float
        Frequency (GHz)
    particle_size : int, float or array
        Particle size a [m].
    diel_constant_p : complex
        Dielectric constant of the medium.
    diel_constant_b : complex
        Dielectric constant of the background.

    Returns
    -------
    All returns are attributes!
    self.ke : int, float or array_like
        Extinction coefficient.
    self.ks : int, float or array_like
        Scattering coefficient.
    self.ka : int, float or array_like
        Absorption coefficient.
    self.om : int, float or array_like
        Omega.
    self.s0 : int, float or array_like
        Backscatter coefficient sigma 0.

    """

    def __init__(self, frequency, particle_size, diel_constant_p, diel_constant_b=(1 + 1j)):

        super(Rayleigh, self).__init__(frequency, particle_size, diel_constant_p, diel_constant_b)

        # Check validity
        lm = 299792458 / (self.freq * 1e9)  # Wavelength in meter
        self.condition = (2 * np.pi * self.a) / lm

        if np.any(self.condition >= 0.5):
            warnings.warn("Rayleigh condition not holds. You should use Mie scattering.", Warning)
        else:
            pass

        self.__calc()

    def __calc(self):
        self.bigK = (self.n ** 2 - 1) / (self.n ** 2 + 2)
        self.ks = (8 / 3) * self.chi ** 4 * np.abs(self.bigK) ** 2
        self.ka = 4 * self.chi * (-self.bigK.imag)
        self.ke = self.ka + self.ks
        self.kt = 1 - self.ke
        self.s0 = 4 * self.chi ** 4 * np.abs(self.bigK) ** 2
        self.omega = self.ks / self.ke


class Mie(Scattering):
    """
    Calculate the extinction coefficients in terms of Mie
    scattering (:cite:`Ulaby.2015` and :cite:`Ulaby.2015b`).

    Parameters
    ----------
    frequency : int or float
        Frequency (GHz)
    particle_size : int, float or array
        Particle size a [m].
    diel_constant_p : complex
        Dielectric constant of the medium.
    diel_constant_b : complex
        Dielectric constant of the background.

    Returns
    -------
    All returns are attributes!
    self.ke : int, float or array_like
        Extinction coefficient.
    self.ks : int, float or array_like
        Scattering coefficient.
    self.ka : int, float or array_like
        Absorption coefficient.
    self.om : int, float or array_like
        Omega.
    self.s0 : int, float or array_like
        Backscatter coefficient sigma 0.
    """

    def __init__(self, frequency, particle_size, diel_constant_p, diel_constant_b=(1 + 1j)):

        super(Mie, self).__init__(frequency, particle_size, diel_constant_p, diel_constant_b)

        # Check validity
        lm = 299792458 / (self.freq * 1e9)  # Wavelength in meter
        self.condition = (2 * np.pi * self.a) / lm

        if np.any(self.condition < 0.5):
            warnings.warn("Mie condition not holds. You schould use Rayleigh scattering.", Warning)
        else:
            pass

        try:
            self.lenchi = len(self.chi)
        except:
            self.lenchi = 1

        self.__calc()

    def __end_sum(self, A0, A1, num):
        stop = True
        pDiff = np.abs((A1 - A0) / A0) * 100

        try:
            for t in srange(num):
                if pDiff[t] >= 0.001 or A0[t] == 0:
                    stop = False
                else:
                    pass
            return stop
        except:
            if pDiff >= 0.001 or A0 == 0:
                stop = False
            else:
                pass
            return stop

    def __calc(self):
        l = 1
        first = True
        runSum = np.zeros_like(self.freq)
        oldSum = np.zeros_like(self.freq)

        W1 = np.sin(self.chi) + 1j * np.cos(self.chi)
        W2 = np.cos(self.chi) - 1j * np.sin(self.chi)
        A1 = cot(self.n * self.chi)

        while first or not self.__end_sum(oldSum, runSum, self.lenchi):
            W = (2 * l - 1) / self.chi * W1 - W2

            A = -l / (self.n * self.chi) + (l / (self.n * self.chi) - A1) ** (-1)

            a = ((A / self.n + l / self.chi) * W.real - W1.real) / ((A / self.n + l / self.chi) * W - W1)
            b = ((self.n * A + l / self.chi) * W.real - W1.real) / ((self.n * A + l / self.chi) * W - W1)

            sumTerm = (2 * l + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2)
            oldSum = runSum
            runSum = runSum + sumTerm

            l += 1

            W2 = W1
            W1 = W

            A1 = A

            first = False

        self.ks = 2 / self.chi ** 2 * runSum

        l = 1
        first = True
        runSum = np.zeros_like(self.freq)
        oldSum = np.zeros_like(self.freq)

        W1 = np.sin(self.chi) + 1j * np.cos(self.chi)
        W2 = np.cos(self.chi) - 1j * np.sin(self.chi)
        A1 = cot(self.n * self.chi)

        while first or not self.__end_sum(oldSum, runSum, self.lenchi):
            W = (2 * l - 1) / self.chi * W1 - W2
            A = -l / (self.n * self.chi) + (l / (self.n * self.chi) - A1) ** (-1)

            a = ((A / self.n + l / self.chi) * W.real - W1.real) / ((A / self.n + l / self.chi) * W - W1)
            b = ((self.n * A + l / self.chi) * W.real - W1.real) / ((self.n * A + l / self.chi) * W - W1)

            sumTerm = (2 * l + 1) * np.real(a + b)
            oldSum = runSum
            runSum = runSum + sumTerm

            l += 1

            W2 = W1
            W1 = W

            A1 = A

            first = False

        self.ke = 2 / self.chi ** 2 * runSum
        self.omega = self.ks / self.ke
        self.ka = self.ke - self.ks
        self.kt = 1 - self.ke

        l = 1
        first = True
        runSum = np.zeros_like(self.freq)
        oldSum = np.zeros_like(self.freq)

        W1 = np.sin(self.chi) + 1j * np.cos(self.chi)
        W2 = np.cos(self.chi) - 1j * np.sin(self.chi)
        A1 = cot(self.n * self.chi)

        while first or not self.__end_sum(oldSum, runSum, self.lenchi):
            W = (2 * l - 1) / self.chi * W1 - W2
            A = -l / (self.n * self.chi) + (l / (self.n * self.chi) - A1) ** (-1)

            a = ((A / self.n + l / self.chi) * W.real - W1.real) / ((A / self.n + l / self.chi) * W - W1)
            b = ((self.n * A + l / self.chi) * W.real - W1.real) / ((self.n * A + l / self.chi) * W - W1)

            sumTerm = (-1) ** l * (2 * l + 1) * (a - b)
            oldSum = runSum
            runSum = runSum + sumTerm

            l += 1

            W2 = W1
            W1 = W

            A1 = A

            first = False

        self.s0 = 1 / self.chi ** 2 * np.abs(runSum) ** 2


# ---- Dielectric Constants ----
class DielConstant:
    """
    Class to calculate the Dielectric Constant of different objects (:cite:`Ulaby.2015` and :cite:`Ulaby.2015b`).

    See Also
    --------
    DielConstant.pureWater
    DielConstant.salineWater
    DielConstant.soil
    DielConstant.vegetation,
    DielConstant.combine

    """

    def __init__(self):
        pass

    @staticmethod
    def water(frequency, temp):
        # <Help and Info Section> -----------------------------------------
        """
        Relative Dielectric Constant of Pure Water.
        Computes the real and imaginary parts of the relative
        dielectric constant of water at any temperature 0<t<30 and frequency
        0<f<1000 GHz. Uses the double-Debye model.

        Parameters
        ----------
        frequency : int, float or array_like
            Frequency (GHz).
        temp : int, float or array
            Temperature in C° (0 - 30).

        Returns
        -------
        Dielectric Constant:    complex

        """
        a = [0.63000075e1, 0.26242021e-2, 0.17667420e-3, 0.58366888e3, 0.12634992e3, 0.69227972e-4, 0.30742330e3,
             0.12634992e3, 0.37245044e1, 0.92609781e-2]

        epsS = 87.85306 * np.exp(-0.00456992 * temp)
        epsOne = a[0] * np.exp(-a[1] * temp)
        tau1 = a[2] * np.exp(a[3] / (temp + a[4]))
        tau2 = a[5] * np.exp(a[6] / (temp + a[7]))
        epsInf = a[8] + a[9] * temp

        eps = ((epsS - epsOne) / (1 - 1j * 2 * np.pi * frequency * tau1)) + (
                (epsOne - epsInf) / (1 - 1j * 2 * np.pi * frequency * tau2)) + epsInf

        return eps

    @staticmethod
    def saline_water(frequency, temp, salinity):
        # <Help and Info Section> -----------------------------------------
        """
        Relative Dielectric Constant of Saline Water.
        Computes the real and imaginary parts of the relative
        dielectric constant of water at any temperature 0<t<30, Salinity
        0<Salinity<40 0/00, and frequency 0<f<1000GHz

        Parameters
        ----------
        frequency : int, float or array_like
            Frequency (GHz).
        temp : int, float or array
            Temperature in C° (0 - 30).
        salinity : int, float or array
            Salinity in parts per thousand.

        Returns
        -------
        Dielectric Constant:    complex

        """
        # Conductvity
        A = [2.903602, 8.607e-2, 4.738817e-4, -2.991e-6, 4.3041e-9]
        sig35 = A[0] + A[1] * temp + A[2] * temp ** 2 + A[3] * temp ** 3 + A[4] * temp ** 4

        A = [37.5109, 5.45216, 0.014409, 1004.75, 182.283]
        P = salinity * ((A[0] + A[1] * salinity + A[2] * salinity ** 2) / (A[3] + A[4] * salinity + salinity ** 2))

        A = [6.9431, 3.2841, -0.099486, 84.85, 69.024]
        alpha0 = (A[0] + A[1] * salinity + A[2] * salinity ** 2) / (A[3] + A[4] * salinity + salinity ** 2)

        A = [49.843, -0.2276, 0.00198]
        alpha1 = A[0] + A[1] * salinity + A[2] * salinity ** 2

        Q = 1 + ((alpha0 * (temp - 15)) / (temp + alpha1))

        sigma = sig35 * P * Q

        a = [0.46606917e-2, -0.26087876e-4, -0.63926782e-5, 0.63000075e1, 0.26242021e-2, -0.42984155e-2, 0.34414691e-4,
             0.17667420e-3, -0.20491560e-6, 0.58366888e3, 0.12634992e3, 0.69227972e-4, 0.38957681e-6, 0.30742330e3,
             0.12634992e3, 0.37245044e1, 0.92609781e-2, -0.26093754e-1]

        epsS = 87.85306 * np.exp(-0.00456992 * temp - a[0] * salinity - a[1] * salinity ** 2 - a[2] * salinity * temp)
        epsOne = a[3] * np.exp(-a[4] * temp - a[5] * salinity - a[6] * salinity * temp)
        tau1 = (a[7] + a[8] * salinity) * np.exp(a[9] / (temp + a[10]))
        tau2 = (a[11] + a[12] * salinity) * np.exp(a[13] / (temp + a[14]))
        epsInf = a[15] + a[16] * temp + a[17] * salinity

        eps = ((epsS - epsOne) / (1 - 1j * 2 * np.pi * frequency * tau1)) + (
                (epsOne - epsInf) / (1 - 1j * 2 * np.pi * frequency * tau2)) + epsInf + 1j * (
                      (17.9751 * sigma) / frequency)

        return eps

    @staticmethod
    def soil(frequency, temp, S, C, mv, rho_b=1.7):
        # <Help and Info Section> -----------------------------------------
        """
        Relative Dielectric Constant of soil.
        Computes the real and imaginary parts of the relative
        dielectric constant of soil at a given temperature 0<t<40C, frequency,
        volumetric moisture content, soil bulk density, sand and clay
        fractions.

        Parameters
        ----------
        frequency : int, float or array_like
            Frequency (GHz).
        temp : int, float or array
            Temperature in C° (0 - 30).
        S : int or float
            Sand fraction in %.
        C : int or float
            Clay fraction in %.
        mv : int or float
            Volumetric Water Content (0<mv<1)
        rho_b : int or float (default = 1.7)
            Bulk density in g/cm3 (typical value is 1.7 g/cm3).

        Returns
        -------
        Dielectric Constant:    complex

        """
        frequency = np.asarray(frequency).flatten()
        epsl = []
        for i in srange(len(frequency)):
            f_hz = frequency[i] * 1.0e9

            beta1 = 1.27 - 0.519 * S - 0.152 * C
            beta2 = 2.06 - 0.928 * S - 0.255 * C
            alpha = 0.65

            eps_0 = 8.854e-12

            sigma_s = 0
            if frequency[i] > 1.3:
                sigma_s = -1.645 + 1.939 * rho_b - 2.256 * S + 1.594 * C

            if frequency[i] >= 0.3 and frequency[i] <= 1.3:
                sigma_s = 0.0467 + 0.22 * rho_b - 0.411 * S + 0.661 * C

            ew_inf = 4.9
            ew_0 = 88.045 - 0.4147 * temp + 6.295e-4 * temp ** 2 + 1.075e-5 * temp ** 3
            tau_w = (1.1109e-10 - 3.824e-12 * temp + 6.938e-14 * temp ** 2 - 5.096e-16 * temp ** 3) / 2 / np.pi

            epsrW = ew_inf + (ew_0 - ew_inf) / (1 + (2 * np.pi * f_hz * tau_w) ** 2)

            epsiW = 2 * np.pi * tau_w * f_hz * (ew_0 - ew_inf) / (1 + (2 * np.pi * f_hz * tau_w) ** 2) + (
                    2.65 - rho_b) / 2.65 / mv * sigma_s / (2 * np.pi * eps_0 * f_hz)

            epsr = (1 + 0.66 * rho_b + mv ** beta1 * epsrW ** alpha - mv) ** (1 / alpha)
            epsi = mv ** beta2 * epsiW

            eps = np.complex(epsr, epsi)
            epsl.append(eps)

        return np.asarray(epsl, dtype=np.complex)

    @staticmethod
    def vegetation(frequency, mg):
        # <Help and Info Section> -----------------------------------------
        """
        Relative Dielectric Constant of Vegetation.
        Computes the real and imaginary parts of the relative
        dielectric constant of vegetation material, such as corn leaves, in
        the microwave region.

        Parameters
        ----------
        frequency : int, float or array_like
            Frequency (GHz).
        mg : int or float
            Gravimetric moisture content (0<mg< 1).

        Returns
        -------
        Dielectric Constant:    complex

        """
        frequency = np.asarray(frequency).flatten()

        S = 15

        epsl = []
        for i in srange(len(frequency)):
            # free water in leaves
            sigma_i = 0.17 * S - 0.0013 * S ** 2

            eps_w_r = 4.9 + 74.4 / (1 + (frequency[i] / 18) ** 2)
            eps_w_i = 74.4 * (frequency[i] / 18) / (1 + (frequency[i] / 18) ** 2) + 18 * sigma_i / frequency[i]

            # bound water in leaves
            eps_b_r = 2.9 + 55 * (1 + np.sqrt(frequency[i] / 0.36)) / (
                    (1 + np.sqrt(frequency[i] / 0.36)) ** 2 + (frequency[i] / 0.36))
            eps_b_i = 55 * np.sqrt(frequency[i] / 0.36) / (
                    (1 + np.sqrt(frequency[i] / 0.36)) ** 2 + (frequency[i] / 0.36))

            # emnp.pirical fits
            v_fw = mg * (0.55 * mg - 0.076)
            v_bw = 4.64 * mg ** 2 / (1 + 7.36 * mg ** 2)

            eps_r = 1.7 - 0.74 * mg + 6.16 * mg ** 2
            eps_v_r = eps_r + v_fw * eps_w_r + v_bw * eps_b_r
            eps_v_i = v_fw * eps_w_i + v_bw * eps_b_i

            eps = np.complex(eps_v_r, eps_v_i)
            epsl.append(eps)

        return np.asarray(epsl, dtype=np.complex)

    @staticmethod
    def combine(frequency, mg, temp, S, C, mv, rho_b=1.7):
        # <Help and Info Section> -----------------------------------------
        """
        Combine the Relative Dielectric Constant of Vegetation with Soil.
        Computes the real and imaginary parts of the relative
        dielectric constant of vegetation material, such as corn leaves, in
        the microwave region.

        Computes the real and imaginary parts of the relative
        dielectric constant of soil at a given temperature 0<t<40C, frequency,
        volumetric moisture content, soil bulk density, sand and clay
        fractions.

        Parameters
        ----------
        frequency : int, float or array_like
            Frequency (GHz).
        mg : int or float
            Gravimetric moisture content (0<mg< 1).
        temp : int, float or array
            Temperature in C° (0 - 30).
        S : int or float
            Sand fraction in %.
        C : int or float
            Clay fraction in %.
        mv : int or float
            Volumetric Water Content (0<mv<1)
        rho_b : int or float (default = 1.7)
            Bulk density in g/cm3 (typical value is 1.7 g/cm3).

        """
        surf = DielConstant.soil(frequency, temp, S, C, mv, rho_b)
        veg = DielConstant.vegetation(frequency, mg)

        return ReflectanceResult(freq=frequency,
                                 surface=surf,
                                 vegetation=veg)


# ---- Correlation Function ----
class CorrFunc:
    """
    Correlation Functions for I2EM Model.

    Parameters
    ----------
    Functions:
                        * class.exponential(n, wvnb, sigma, corrlength, Ts),
                        * class.gaussian(n, wvnb, sigma, corrlength, Ts),
                        * class.xpower(n, wvnb, sigma, corrlength, Ts),

    n:      int (>1)
                        Coefficient needed for x-power and x-exponential
                        correlation function

    wvnb:
                        Calculated by SurfScat Module.

    corrlength:         int or floar)
                        Correlation length (cm)

    sigma:  int or float)
                        RMS Height (cm)

    Ts:
                        Calculated by SurfScat Module.

    """

    def __init__(self):
        pass

    def calc(self):
        raise NotImplementedError("Subclass must implement abstract method")


class exponential(CorrFunc):
    """
    See Also
    --------
    CorrFunc
    """

    def __init__(self, n, wvnb, sigma, corrlength, Ts):
        self.n = n
        self.wvnb = wvnb
        self.sigma = sigma
        self.corrlen = corrlength
        self.Ts = Ts
        self.calc()

    def calc(self):
        Wn = []
        for i in srange(self.Ts):
            i += 1
            self.wn = self.corrlen ** 2 / i ** 2 * (1 + (self.wvnb * self.corrlen / i) ** 2) ** (-1.5)
            Wn.append(self.wn)

        self.Wn = np.asarray(Wn, dtype=np.float)
        self.rss = self.sigma / self.corrlen


class gaussian(CorrFunc):
    """
    See Also
    --------
    CorrFunc
    """

    def __init__(self, n, wvnb, sigma, corrlength, Ts):
        self.n = n
        self.wvnb = wvnb
        self.sigma = sigma
        self.corrlen = corrlength
        self.Ts = Ts
        self.calc()

    def calc(self):
        Wn = []
        for i in srange(self.Ts):
            i += 1
            self.wn = self.corrlen ** 2 / (2 * i) * np.exp(-(self.wvnb * self.corrlen) ** 2 / (4 * i))
            Wn.append(self.wn)

        self.Wn = np.asarray(Wn, dtype=np.float)
        self.rss = np.sqrt(2) * self.sigma / self.corrlen


class xpower(CorrFunc):
    """
    See Also
    --------
    CorrFunc
    """

    def __init__(self, n, wvnb, sigma, corrlength, Ts):
        self.n = n
        self.wvnb = wvnb
        self.sigma = sigma
        self.corrlen = corrlength
        self.Ts = Ts
        self.calc()

    def calc(self):
        import scipy as sp
        Wn = []
        for i in srange(self.Ts):
            i += 1
            self.wn = self.corrlen ** 2 * (self.wvnb * self.corrlen) ** (-1 + self.n * i) * sp.special.kv(
                1 - self.n * i, self.wvnb * self.corrlen) / (2 ** (self.n * i - 1) * sp.special.gamma(self.n * i))
            Wn.append(self.wn)

        self.Wn = np.asarray(Wn, dtype=np.float)
        if self.n == 1.5:
            self.rss = np.sqrt(self.n * 2) * self.sigma / self.corrlen
        else:
            self.rss = 0


class mixed(CorrFunc):
    def __init__(self, n, wvnb, sigma, corrlength, Ts):
        self.n = n
        self.wvnb = wvnb
        self.sigma = sigma
        self.corrlen = corrlength
        self.Ts = Ts
        self.calc()

    def calc(self):
        gauss = gaussian(self.n, self.wvnb, self.sigma, self.corrlen, self.Ts)
        exp = exponential(self.n, self.wvnb, self.sigma, self.corrlen, self.Ts)

        self.Wn = gauss.Wn / exp.Wn
        self.rss = gauss.rss / exp.rss


# ---- Surface Models ----
class LSM:
    """
    In optical wavelengths the Lambertian Linear Model (LSM) is used. If you
    want to calculate the RO Model in optical terms you will calculate the
    surface reflexion previous.

    Equation:
    Total Soil Reflectance = Reflectance*(Moisture*soil_spectrum1+(1-Moisture)*soil_spectrum2)

    By default, soil_spectrum1 is a dry soil, and soil_spectrum2 is a
    wet soil, so in that case, 'moisture' is a surface soil moisture parameter.
    ``reflectance`` is a  soil brightness term. You can provide one or the two
    soil spectra if you want.  The calculation is between 400 and 2500 nm with
    1nm spacing.

    Parameters
    ----------
    reflectance : int or float
        Surface (Lambertian) reflectance in optical wavelength.
    moisture : int or float
        Surface moisture content between 0 and 1.

    Returns
    -------
    All returns are attributes!
    self.L8 : namedtuple (with dot access)
        Landsat 8 average kx (ks, kt, ke) values for Bx band (B2 until B7)
    self.ASTER : namedtuple (with dot access)
        ASTER average kx (ks, kt, ke) values for Bx band (B1 until B9)
    self.ref : dict (with dot access)
        Continuous surface reflectance values from 400 until 2500 nm
    self.l : dict (with dot access)
        Continuous Wavelength values from 400 until 2500 nm


    """

    def __init__(self, reflectance, moisture):

        self.l = np.arange(400, 2501)
        self.sRef = reflectance
        self.moisture = moisture
        self.__calc()
        self.__store()

    def __calc(self):
        self.ref = self.sRef * (self.moisture * lib.soil.rsoil1 + (1 - self.moisture) * lib.soil.rsoil2)
        self.int = [self.l, self.ref]
        self.int = np.asarray(self.int, dtype=np.float32)
        self.int = self.int.transpose()

    #        self.surface = ReflectanceResult(ref=self.ref,
    #       l=self.l)

    def __store(self):
        # <Help and Info Section> -----------------------------------------
        """
        Store the surface reflectance for ASTER bands B1 - B9 or LANDSAT8 bands
        B2 - B7.

        Access:
            :self.ASTER.Bx:     (array_like)
                                Soil reflectance for ASTER Band x.
            :self.L8.Bx:        (array_like)
                                Soil reflectance for LANDSAT 8 Band x.
        """

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

        B1 = self.int[(self.int[:, 0] >= b1[0]) & (self.int[:, 0] <= b1[1])]
        B1 = B1[:, 1].mean()

        B2 = self.int[(self.int[:, 0] >= b2[0]) & (self.int[:, 0] <= b2[1])]
        B2 = B2[:, 1].mean()

        B3 = self.int[(self.int[:, 0] >= b3[0]) & (self.int[:, 0] <= b3[1])]
        B3 = B3[:, 1].mean()

        B4 = self.int[(self.int[:, 0] >= b4[0]) & (self.int[:, 0] <= b4[1])]
        B4 = B4[:, 1].mean()

        B5 = self.int[(self.int[:, 0] >= b5[0]) & (self.int[:, 0] <= b5[1])]
        B5 = B5[:, 1].mean()

        B6 = self.int[(self.int[:, 0] >= b6[0]) & (self.int[:, 0] <= b6[1])]
        B6 = B6[:, 1].mean()

        B7 = self.int[(self.int[:, 0] >= b7[0]) & (self.int[:, 0] <= b7[1])]
        B7 = B7[:, 1].mean()

        B8 = self.int[(self.int[:, 0] >= b8[0]) & (self.int[:, 0] <= b8[1])]
        B8 = B8[:, 1].mean()

        B9 = self.int[(self.int[:, 0] >= b9[0]) & (self.int[:, 0] <= b9[1])]
        B9 = B9[:, 1].mean()

        self.ASTER = ASTER(B1, B2, B3, B4, B5, B6, B7, B8, B9)

        L8 = namedtuple('L8', 'B2 B3 B4 B5 B6 B7')

        b2 = (452, 452 + 60)
        b3 = (533, 533 + 57)
        b4 = (636, 636 + 37)
        b5 = (851, 851 + 28)
        b6 = (1566, 1566 + 85)
        b7 = (2107, 2107 + 187)

        B2 = self.int[(self.int[:, 0] >= b2[0]) & (self.int[:, 0] <= b2[1])]
        B2 = B2[:, 1].mean()

        B3 = self.int[(self.int[:, 0] >= b3[0]) & (self.int[:, 0] <= b3[1])]
        B3 = B3[:, 1].mean()

        B4 = self.int[(self.int[:, 0] >= b4[0]) & (self.int[:, 0] <= b4[1])]
        B4 = B4[:, 1].mean()

        B5 = self.int[(self.int[:, 0] >= b5[0]) & (self.int[:, 0] <= b5[1])]
        B5 = B5[:, 1].mean()

        B6 = self.int[(self.int[:, 0] >= b6[0]) & (self.int[:, 0] <= b6[1])]
        B6 = B6[:, 1].mean()

        B7 = self.int[(self.int[:, 0] >= b7[0]) & (self.int[:, 0] <= b7[1])]
        B7 = B7[:, 1].mean()

        self.L8 = L8(B2, B3, B4, B5, B6, B7)

    #
    #        self.int = ReflectanceResult(L8=L8,
    #   ASTER=ASTER,
    #   surface=self.surface)

    def select(self, mins, maxs, function='mean'):
        # <Help and Info Section> -----------------------------------------
        """
        Returns the means of the coefficients in range between min and max.

        Args:
            :min:   (int)
                                Lower bound of the wavelength (400 - 2500)

            :max:   (int)
                                Upper bound of the wavelength (400 - 2500)

        """
        if function == 'mean':
            ranges = self.int[(self.int[:, 0] >= mins) & (self.int[:, 0] <= maxs)]
            ref = ranges[:, 1].mean()

            return ref

    def cleanup(self, name):
        """Do cleanup for an attribute"""
        try:
            delattr(self, name)
        except TypeError:
            for item in name:
                delattr(self, item)


class I2EM(Kernel):
    """
     RADAR Surface Scatter Based Kernel (I2EM). Compute BSC VV and
     BSC HH and the emissivity for single-scale random surface for
     Bi and Mono-static acquisitions (:cite:`Ulaby.2015` and :cite:`Ulaby.2015b`).

     Parameters
     ----------
     iza, vza, raa : int, float or ndarray
         Incidence (iza) and scattering (vza) zenith angle, as well as relative azimuth (raa) angle.
     normalize : boolean, optional
         Set to 'True' to make kernels 0 at nadir view illumination. Since all implemented kernels are normalized
         the default value is False.
     nbar : float, optional
         The sun or incidence zenith angle at which the isotropic term is set
         to if normalize is True. The default value is 0.0.
     angle_unit : {'DEG', 'RAD'}, optional
         * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
         * 'RAD': All input angles (iza, vza, raa) are in [RAD].
     frequency : int or float
         RADAR Frequency (GHz).
     diel_constant : int or float
         Complex dielectric constant of soil.
     corrlength : int or float
         Correlation length (cm).
     sigma : int or float
         RMS Height (cm)
     n : int (default = 10), optinal
         Coefficient needed for x-power and x-exponential
         correlation function.
     corrfunc : {'exponential', 'gaussian', 'xpower', 'mixed'}, optional
         Correlation distribution functions. The `mixed` correlation function is the result of the division of
         gaussian correlation function with exponential correlation function. Default is 'exponential'.

     Returns
     -------
     For more attributes see also pyrism.core.Kernel and pyrism.core.ReflectanceResult.

     See Also
     --------
     I2EM.Emissivity
     pyrism.core.Kernel
     pyrism.core.ReflectanceResult


     Note
     ----
     The model is constrained to realistic surfaces with
     (rms height / correlation length) ≤ 0.25.
     Hot spot direction is vza == iza and raa = 0.0

     """

    # TODO: Delete unnecessary self. calls.

    def __init__(self, iza, vza, raa, normalize=True, nbar=0.0, angle_unit='DEG', frequency=None, diel_constant=None,
                 corrlength=None, sigma=None, n=10, corrfunc='exponential'):

        super(I2EM, self).__init__(iza, vza, raa, normalize, nbar, angle_unit)

        if corrfunc is 'exponential':
            self.corrfunc = exponential
        elif corrfunc is 'gaussian':
            self.corrfunc = gaussian
        elif corrfunc is 'xpower':
            self.corrfunc = xpower
        elif corrfunc is 'mixed':
            self.corrfunc = mixed
        else:
            raise ValueError("The parameter corrfunc must be 'exponential', 'gaussian' or 'xpower'")

        self.er = diel_constant
        self.corrlen = corrlength  # in cm
        self.n = n
        self.sigma = sigma  # in cm
        self.freq = frequency

        self.__set_coef()
        self.__reflection_coefficients()
        self.__r_transition()
        self.__average_reflection_coefficients()
        self.__biStatic_coefficient()
        self.__Ipp()
        self.__shadowing_function()
        self.__sigma_nought()
        self.__normalize()
        self.__store()

    def __normalize(self):
        self.norm = 0.
        # if we are normalising the last element of self.Isotropic, self.Ross and self.Li contain
        # the nadir-nadir kernel
        if self.normalize == True:
            # normalize nbar-nadir (so kernel is 0 at nbar-nadir)
            self.norm = self.VV[-1]

            # depreciate length of arrays (well, teh ones we'll use again in any case)
            self.VV = self.VV[0:-1]
            self.HH = self.HH[0:-1]
            self.VVdB = self.VVdB[0:-1]
            self.HHdB = self.HHdB[0:-1]

            self.vzaDeg = self.vzaDeg[0:-1]
            self.izaDeg = self.izaDeg[0:-1]
            self.raaDeg = self.raaDeg[0:-1]
            self.N = len(self.vzaDeg)
            self.vza = self.vza[0:-1]
            self.iza = self.iza[0:-1]
            self.raa = self.raa[0:-1]

    def __set_coef(self):
        self.phi = 0
        self.merror = 1.0e8
        self.k = 2 * np.pi * self.freq / 30
        self.kz_iza = self.k * np.cos(self.iza + 0.01)
        self.kz_vza = self.k * np.cos(self.vza)

    def __reflection_coefficients(self):
        warnings.filterwarnings("ignore")

        self.rt = np.sqrt(self.er - np.sin(self.iza + 0.01) ** 2)
        self.Rvi = (self.er * np.cos(self.iza + 0.01) - self.rt) / (self.er * np.cos(self.iza + 0.01) + self.rt)
        self.Rhi = (np.cos(self.iza + 0.01) - self.rt) / (np.cos(self.iza + 0.01) + self.rt)
        self.wvnb = self.k * np.sqrt(
            (np.sin(self.vza) * np.cos(self.raa) - np.sin(self.iza + 0.01) * np.cos(self.phi)) ** 2 + (
                    np.sin(self.vza) * np.sin(self.raa) - np.sin(self.iza + 0.01) * np.sin(self.phi)) ** 2)
        self.Ts = 1

        while self.merror >= 1.0e-3 and self.Ts <= 150:
            self.Ts += 1
            self.error = ((self.k * self.sigma) ** 2 * (
                    np.cos(self.iza + 0.01) + np.cos(self.vza)) ** 2) ** self.Ts / factorial(self.Ts)
            self.merror = self.error.mean()

        self.CorrFunc = self.corrfunc(self.n, self.wvnb, self.sigma, self.corrlen, self.Ts)

    def __r_transition(self):
        warnings.filterwarnings("ignore")
        self.Rv0 = (np.sqrt(self.er) - 1) / (np.sqrt(self.er) + 1)
        self.Rh0 = -self.Rv0

        self.Ft = 8 * self.Rv0 ** 2 * np.sin(self.vza) * (
                np.cos(self.iza + 0.01) + np.sqrt(self.er - np.sin(self.iza + 0.01) ** 2)) / (
                          np.cos(self.iza + 0.01) * np.sqrt(self.er - np.sin(self.iza + 0.01) ** 2))
        self.a1 = 0
        self.b1 = 0

        for i in srange(self.Ts):
            i += 1
            self.a0 = ((self.k * self.sigma) * np.cos(self.iza + 0.01)) ** (2 * i) / factorial(i)
            self.a1 = self.a1 + self.a0 * self.CorrFunc.Wn[i - 1]
            self.b1 = self.b1 + self.a0 * (np.abs(
                self.Ft / 2 + 2 ** (i + 1) * self.Rv0 / np.cos(self.iza + 0.01) * np.exp(
                    - ((self.k * self.sigma) * np.cos(self.iza + 0.01)) ** 2))) ** 2 * self.CorrFunc.Wn[i - 1]

        self.St = 0.25 * (np.abs(self.Ft) ** 2) * self.a1 / self.b1
        self.St0 = 1 / (np.abs(1 + 8 * self.Rv0 / (np.cos(self.iza + 0.01) * self.Ft))) ** 2
        self.Tf = 1 - self.St / self.St0

    def __average_reflection_coefficients(self):
        # <Help and Info Section> -----------------------------------------
        # Calculate the average reflection coefficients.  These coefficients
        # account for slope effects, especially near the brewster angle.  They are
        # not important if the slope is small.

        warnings.filterwarnings("ignore")

        self.sigx = 1.1 * self.sigma / self.corrlen
        self.sigy = self.sigx
        self.xxx = 3 * self.sigx

        def RaV_integration():
            warnings.filterwarnings("ignore")
            rav = []
            for i in srange(len(self.iza)):
                def integration(Zy, Zx):
                    self.A = np.cos(self.iza + 0.01)[i] + Zx * np.sin(self.iza + 0.01)[i]
                    self.B = self.er * (1 + Zx ** 2 + Zy ** 2)
                    self.CC = np.sin(self.iza + 0.01)[i] ** 2 - 2 * Zx * np.sin(self.iza + 0.01)[i] * \
                              np.cos(self.iza + 0.01)[i] + Zx ** 2 * np.cos(self.iza + 0.01)[i] ** 2 + Zy ** 2
                    self.Rv = (self.er * self.A - np.sqrt(self.B - self.CC)) / (
                            self.er * self.A + np.sqrt(self.B - self.CC))
                    self.pd = np.exp(-Zx ** 2 / (2 * self.sigx ** 2) - Zy ** 2 / (2 * self.sigy ** 2))
                    Rav = self.Rv * self.pd
                    return Rav

                ravv = dblquad(integration, -self.xxx, self.xxx, lambda x: -self.xxx, lambda x: self.xxx)
                temp = np.asarray(ravv[0]) / (2 * np.pi * self.sigx * self.sigy)
                rav.append(temp)
            self.Rav = np.asarray(rav)
            return self.Rav

        def RaH_integration():
            warnings.filterwarnings("ignore")

            rah = []
            for i in srange(len(self.iza)):
                def integration(Zy, Zx):
                    self.A = np.cos(self.iza + 0.01)[i] + Zx * np.sin(self.iza + 0.01)[i]
                    self.B = self.er * (1 + Zx ** 2 + Zy ** 2)
                    self.CC = np.sin(self.iza + 0.01)[i] ** 2 - 2 * Zx * np.sin(self.iza + 0.01)[i] * \
                              np.cos(self.iza + 0.01)[i] + Zx ** 2 * np.cos(self.iza + 0.01)[i] ** 2 + Zy ** 2

                    self.Rh = (self.A - np.sqrt(self.B - self.CC)) / (self.A + np.sqrt(self.B - self.CC))

                    self.pd = np.exp(-Zx ** 2 / (2 * self.sigx ** 2) - Zy ** 2.0 / (2 * self.sigy ** 2))
                    RaH = self.Rh * self.pd
                    return RaH

                rahh = dblquad(integration, -self.xxx, self.xxx, lambda x: -self.xxx, lambda x: self.xxx)
                temp = np.asarray(rahh[0]) / (2 * np.pi * self.sigx * self.sigy)
                rah.append(temp)
            self.Rah = np.asarray(rah)
            return self.Rah

        self.Rav = RaV_integration()
        self.Rah = RaH_integration()

    def __biStatic_coefficient(self):
        warnings.filterwarnings("ignore")

        if np.array_equal(self.vza, self.iza) == True and (np.all(self.raa) == 3.14159265) == True:
            self.Rvt = self.Rvi + (self.Rv0 - self.Rvi) * self.Tf
            self.Rht = self.Rhi + (self.Rh0 - self.Rhi) * self.Tf

        else:
            self.Rvt = self.Rav
            self.Rht = self.Rah

        self.fvv = 2 * self.Rvt * (
                np.sin(self.iza + 0.01) * np.sin(self.vza) - (1 + np.cos(self.iza + 0.01) * np.cos(self.vza)) * np.cos(
            self.raa)) / (np.cos(self.iza + 0.01) + np.cos(self.vza))
        self.fhh = -2 * self.Rht * (
                np.sin(self.iza + 0.01) * np.sin(self.vza) - (1 + np.cos(self.iza + 0.01) * np.cos(self.vza)) * np.cos(
            self.raa)) / (np.cos(self.iza + 0.01) + np.cos(self.vza))

    def __Fppupdn_calc(self, ud, method, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs):
        warnings.filterwarnings("ignore")

        if method == 1:
            Gqi = ud * kz
            Gqti = ud * k * np.sqrt(er - s ** 2)
            qi = ud * kz

            c11 = k * cfs * (ksz - qi)
            c21 = cs * (cfs * (
                    k ** 2 * s * cf * (ss * cfs - s * cf) + Gqi * (k * css - qi)) + k ** 2 * cf * s * ss * sfs ** 2)
            c31 = k * s * (s * cf * cfs * (k * css - qi) - Gqi * (cfs * (ss * cfs - s * cf) + ss * sfs ** 2))
            c41 = k * cs * (cfs * css * (k * css - qi) + k * ss * (ss * cfs - s * cf))
            c51 = Gqi * (cfs * css * (qi - k * css) - k * ss * (ss * cfs - s * cf))

            c12 = k * cfs * (ksz - qi)
            c22 = cs * (cfs * (
                    k ** 2 * s * cf * (ss * cfs - s * cf) + Gqti * (k * css - qi)) + k ** 2 * cf * s * ss * sfs ** 2)
            c32 = k * s * (s * cf * cfs * (k * css - qi) - Gqti * (cfs * (ss * cfs - s * cf) - ss * sfs ** 2))
            c42 = k * cs * (cfs * css * (k * css - qi) + k * ss * (ss * cfs - s * cf))
            c52 = Gqti * (cfs * css * (qi - k * css) - k * ss * (ss * cfs - s * cf))

        if method == 2:
            Gqs = ud * ksz
            Gqts = ud * k * np.sqrt(er - ss ** 2)
            qs = ud * ksz

            c11 = k * cfs * (kz + qs)
            c21 = Gqs * (cfs * (cs * (k * cs + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs ** 2)
            c31 = k * ss * (k * cs * (ss * cfs - s * cf) + s * (kz + qs))
            c41 = k * css * (cfs * (cs * (kz + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs ** 2)
            c51 = -css * (k ** 2 * ss * (ss * cfs - s * cf) + Gqs * cfs * (kz + qs))

            c12 = k * cfs * (kz + qs)
            c22 = Gqts * (cfs * (cs * (kz + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs ** 2)
            c32 = k * ss * (k * cs * (ss * cfs - s * cf) + s * (kz + qs))
            c42 = k * css * (cfs * (cs * (kz + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs ** 2)
            c52 = -css * (k ** 2 * ss * (ss * cfs - s * cf) + Gqts * cfs * (kz + qs))

        q = kz
        qt = k * np.sqrt(er - s ** 2)

        vv = (1 + Rvi) * (-(1 - Rvi) * c11 / q + (1 + Rvi) * c12 / qt) + \
             (1 - Rvi) * ((1 - Rvi) * c21 / q - (1 + Rvi) * c22 / qt) + \
             (1 + Rvi) * ((1 - Rvi) * c31 / q - (1 + Rvi) * c32 / er / qt) + \
             (1 - Rvi) * ((1 + Rvi) * c41 / q - er * (1 - Rvi) * c42 / qt) + \
             (1 + Rvi) * ((1 + Rvi) * c51 / q - (1 - Rvi) * c52 / qt)

        hh = (1 + Rhi) * ((1 - Rhi) * c11 / q - er * (1 + Rhi) * c12 / qt) - \
             (1 - Rhi) * ((1 - Rhi) * c21 / q - (1 + Rhi) * c22 / qt) - \
             (1 + Rhi) * ((1 - Rhi) * c31 / q - (1 + Rhi) * c32 / qt) - \
             (1 - Rhi) * ((1 + Rhi) * c41 / q - (1 - Rhi) * c42 / qt) - \
             (1 + Rhi) * ((1 + Rhi) * c51 / q - (1 - Rhi) * c52 / qt)

        return vv, hh

    def __Ipp(self):
        warnings.filterwarnings("ignore")

        self.Fvvupi, self.Fhhupi = self.__Fppupdn_calc(+1, 1,
                                                       self.Rvi,
                                                       self.Rhi,
                                                       self.er,
                                                       self.k,
                                                       self.kz_iza,
                                                       self.kz_vza,
                                                       np.sin(self.iza + 0.01),
                                                       np.cos(self.iza + 0.01),
                                                       np.sin(self.vza),
                                                       np.cos(self.vza),
                                                       np.cos(self.phi),
                                                       np.cos(self.raa),
                                                       np.sin(self.raa))

        self.Fvvups, self.Fhhups = self.__Fppupdn_calc(+1, 2,
                                                       self.Rvi,
                                                       self.Rhi,
                                                       self.er,
                                                       self.k,
                                                       self.kz_iza,
                                                       self.kz_vza,
                                                       np.sin(self.iza + 0.01),
                                                       np.cos(self.iza + 0.01),
                                                       np.sin(self.vza),
                                                       np.cos(self.vza),
                                                       np.cos(self.phi),
                                                       np.cos(self.raa),
                                                       np.sin(self.raa))

        self.Fvvdni, self.Fhhdni = self.__Fppupdn_calc(-1, 1,
                                                       self.Rvi,
                                                       self.Rhi,
                                                       self.er,
                                                       self.k,
                                                       self.kz_iza,
                                                       self.kz_vza,
                                                       np.sin(self.iza + 0.01),
                                                       np.cos(self.iza + 0.01),
                                                       np.sin(self.vza),
                                                       np.cos(self.vza),
                                                       np.cos(self.phi),
                                                       np.cos(self.raa),
                                                       np.sin(self.raa))

        self.Fvvdns, self.Fhhdns = self.__Fppupdn_calc(-1, 2,
                                                       self.Rvi,
                                                       self.Rhi,
                                                       self.er,
                                                       self.k,
                                                       self.kz_iza,
                                                       self.kz_vza,
                                                       np.sin(self.iza + 0.01),
                                                       np.cos(self.iza + 0.01),
                                                       np.sin(self.vza),
                                                       np.cos(self.vza),
                                                       np.cos(self.phi),
                                                       np.cos(self.raa),
                                                       np.sin(self.raa))

        self.qi = self.k * np.cos(self.iza + 0.01)
        self.qs = self.k * np.cos(self.vza)

        ivv = []
        ihh = []
        for i in srange(self.Ts):
            i += 1
            self.Ivv = (self.kz_iza + self.kz_vza) ** i * self.fvv * np.exp(
                -self.sigma ** 2 * self.kz_iza * self.kz_vza) + \
                       0.25 * (self.Fvvupi * (self.kz_vza - self.qi) ** (i - 1) * np.exp(
                -self.sigma ** 2 * (self.qi ** 2 - self.qi * (self.kz_vza - self.kz_iza))) + self.Fvvdni * (
                                       self.kz_vza + self.qi) ** (i - 1) * np.exp(
                -self.sigma ** 2 * (self.qi ** 2 + self.qi * (self.kz_vza - self.kz_iza))) + self.Fvvups * (
                                       self.kz_iza + self.qs) ** (i - 1) * np.exp(
                -self.sigma ** 2 * (self.qs ** 2 - self.qs * (self.kz_vza - self.kz_iza))) + self.Fvvdns * (
                                       self.kz_iza - self.qs) ** (i - 1) * np.exp(
                -self.sigma ** 2 * (self.qs ** 2 + self.qs * (self.kz_vza - self.kz_iza))))

            self.Ihh = (self.kz_iza + self.kz_vza) ** i * self.fhh * np.exp(
                -self.sigma ** 2 * self.kz_iza * self.kz_vza) + \
                       0.25 * (self.Fhhupi * (self.kz_vza - self.qi) ** (i - 1) * np.exp(
                -self.sigma ** 2 * (self.qi ** 2 - self.qi * (self.kz_vza - self.kz_iza))) +
                               self.Fhhdni * (self.kz_vza + self.qi) ** (i - 1) * np.exp(
                        -self.sigma ** 2 * (self.qi ** 2 + self.qi * (self.kz_vza - self.kz_iza))) +
                               self.Fhhups * (self.kz_iza + self.qs) ** (i - 1) * np.exp(
                        -self.sigma ** 2 * (self.qs ** 2 - self.qs * (self.kz_vza - self.kz_iza))) +
                               self.Fhhdns * (self.kz_iza - self.qs) ** (i - 1) * np.exp(
                        -self.sigma ** 2 * (self.qs ** 2 + self.qs * (self.kz_vza - self.kz_iza))))

            ivv.append(self.Ivv)
            ihh.append(self.Ihh)
        self.Ivv = np.asarray(ivv, dtype=np.complex)
        self.Ihh = np.asarray(ihh, dtype=np.complex)

    def __shadowing_function(self):
        import scipy as sp
        warnings.filterwarnings("ignore")

        if np.array_equal(self.vza, self.iza) == True and (np.all(self.raa) == 3.14159265) == True:
            ct = cot(self.iza)
            cts = cot(self.vza)
            rslp = self.CorrFunc.rss
            ctorslp = ct / np.sqrt(2) / rslp
            ctsorslp = cts / np.sqrt(2) / rslp
            shadf = 0.5 * (np.exp(-ctorslp ** 2) / np.sqrt(np.pi) / ctorslp - sp.erf(ctorslp))
            shadfs = 0.5 * (np.exp(-ctsorslp ** 2) / np.sqrt(np.pi) / ctsorslp - sp.erf(ctsorslp))
            self.ShdwS = 1 / (1 + shadf + shadfs)
        else:
            self.ShdwS = 1

    def __sigma_nought(self):
        warnings.filterwarnings("ignore")

        self.sigmavv = 0
        self.sigmahh = 0
        for i in srange(self.Ts):
            i += 1
            self.a0 = self.CorrFunc.Wn[i - 1] / factorial(i) * self.sigma ** (2 * i)

            self.sigmavv = self.sigmavv + np.abs(self.Ivv[i - 1]) ** 2 * self.a0
            self.sigmahh = self.sigmahh + np.abs(self.Ihh[i - 1]) ** 2 * self.a0

        self.VV = self.sigmavv * self.ShdwS * self.k ** 2 / 2 * np.exp(
            -self.sigma ** 2 * (self.kz_iza ** 2 + self.kz_vza ** 2))
        self.HH = self.sigmahh * self.ShdwS * self.k ** 2 / 2 * np.exp(
            -self.sigma ** 2 * (self.kz_iza ** 2 + self.kz_vza ** 2))

        with np.errstate(invalid='ignore'):
            self.VVdB = dB(np.asarray(self.VV, dtype=np.float))
            self.HHdB = dB(np.asarray(self.HH, dtype=np.float))

    def __store(self):
        self.BSC = ReflectanceResult(array=np.array([self.VV[0], self.HH[0]]),
                                     arraydB=np.array([dB(self.VV[0]), dB(self.HH[0])]),
                                     VV=self.VV,
                                     HH=self.HH,
                                     VVdB=self.VVdB,
                                     HHdB=self.HHdB)

        self.BRDF = ReflectanceResult(
            array=np.array([BRDF(self.VV, self.iza, self.vza)[0], BRDF(self.HH, self.iza, self.vza)[0]]).flatten(),
            arraydB=np.array(
                [dB(BRDF(self.VV, self.iza, self.vza))[0], dB(BRDF(self.HH, self.iza, self.vza))[0]]).flatten(),
            VV=BRDF(self.VV, self.iza, self.vza),
            HH=BRDF(self.HH, self.iza, self.vza),
            VVdB=dB(BRDF(self.VV, self.iza, self.vza)),
            HHdB=dB(BRDF(self.HH, self.iza, self.vza)))

        self.BRF = ReflectanceResult(array=np.array([BRF(self.BRDF.VV)[0], BRF(self.BRDF.HH)[0]]).flatten(),
                                     arraydB=np.array([dB(BRF(self.BRDF.VV))[0], dB(BRF(self.BRDF.HH))[0]]).flatten(),
                                     VV=BRF(self.BRDF.VV),
                                     HH=BRF(self.BRDF.HH),
                                     VVdB=dB(BRF(self.BRDF.VV)),
                                     HHdB=dB(BRF(self.BRDF.HH)))

    class Emissivity(Kernel):
        """
        This Class calculates the emission from rough surfaces using the
        I2EM Model.

        Parameters
        ----------
         iza, vza, raa : int, float or ndarray
             Incidence (iza) and scattering (vza) zenith angle, as well as relative azimuth (raa) angle.
         normalize : boolean, optional
             Set to 'True' to make kernels 0 at nadir view illumination. Since all implemented kernels are normalized
             the default value is False.
         nbar : float, optional
             The sun or incidence zenith angle at which the isotropic term is set
             to if normalize is True. The default value is 0.0.
         angle_unit : {'DEG', 'RAD'}, optional
             * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
             * 'RAD': All input angles (iza, vza, raa) are in [RAD].
         frequency : int or float
             RADAR Frequency (GHz).
         diel_constant : int or float
             Complex dielectric constant of soil.
         corrlength : int or float
             Correlation length (cm).
         sigma : int or float
             RMS Height (cm)
         corrfunc : {'exponential', 'gaussian', 'mixed'}, optional
             Correlation distribution functions. The `mixed` correlation function is the result of the division of
             gaussian correlation function with exponential correlation function. Default is 'exponential'.

        Returns
        -------
        For attributes see also core.Kernel and core.EmissivityResult.

        See Also
        --------
        pyrism.core.EmissivityResult
        """

        def __init__(self, iza, vza, raa, normalize=False, nbar=0.0, angle_unit='DEG',
                     frequency=1.26, diel_constant=10 + 1j, corrlength=10, sigma=0.3, corrfunc='exponential'):

            super(I2EM.Emissivity, self).__init__(iza, vza, raa, normalize, nbar, angle_unit)

            self.diel_constant = diel_constant
            self.corrlen = corrlength  # in cm
            self.sigma = sigma  # in cm
            self.freq = frequency

            self.corrfunc = corrfunc

            self.__pre_process()
            self.__calc()
            self.__store()

        def __pre_process(self):

            fr = self.freq / 1e9

            self.k = 2 * np.pi * fr / 30  # wavenumber in free space.  Speed of light is in cm/sec

            self.ks = self.k * self.sigma  # roughness parameter
            self.kl = self.k * self.corrlen

            # -- calculation of reflection coefficients
            self.sq = np.sqrt(self.diel_constant - np.sin(self.iza) ** 2)

            self.rv = (self.diel_constant * np.cos(self.iza) - self.sq) / (
                    self.diel_constant * np.cos(self.iza) + self.sq)
            self.rh = (np.cos(self.iza) - self.sq) / (np.cos(self.iza) + self.sq)

        def __calc(self):

            self.pol = 'vv'
            refv = dblquad(self.emsv_integralfunc, 0, np.pi / 2, lambda x: 0, lambda x: np.pi)

            self.pol = 'hh'
            refh = dblquad(self.emsv_integralfunc, 0, np.pi / 2, lambda x: 0, lambda x: np.pi)

            self.VV = 1 - refv[0] - np.exp(-self.ks ** 2 * np.cos(self.iza) * np.cos(self.iza)) * (
                abs(self.rv)) ** 2
            self.HH = 1 - refh[0] - np.exp(-self.ks ** 2 * np.cos(self.iza) * np.cos(self.iza)) * (
                abs(self.rh)) ** 2

            self.VVdB = dB(self.VV)
            self.HHdB = dB(self.HH)

        def __store(self):
            self.EMN = EmissivityResult(array=np.array(
                [(1 - self.VV) / (np.cos(self.iza) * np.cos(self.vza) * 4 * np.pi),
                 (1 - self.HH) / (np.cos(self.iza) * np.cos(self.vza) * 4 * np.pi)]).flatten(),
                                        arraydB=np.array(
                                            [dB((1 - self.VV) / (np.cos(self.iza) * np.cos(self.vza) * 4 * np.pi)),
                                             dB((1 - self.HH) / (
                                                     np.cos(self.iza) * np.cos(self.vza) * 4 * np.pi))]).flatten(),
                                        VV=(1 - self.VV) / (np.cos(self.iza) * np.cos(self.vza) * 4 * np.pi),
                                        HH=(1 - self.HH) / (np.cos(self.iza) * np.cos(self.vza) * 4 * np.pi),
                                        VVdB=dB((1 - self.VV) / (np.cos(self.iza) * np.cos(self.vza) * 4 * np.pi)),
                                        HHdB=dB((1 - self.HH) / (np.cos(self.iza) * np.cos(self.vza) * 4 * np.pi)))

            self.EMS = EmissivityResult(array=np.array([self.VV, self.HH]).flatten(),
                                        arraydB=np.array([dB(self.VV), dB(self.HH)]).flatten(),
                                        VV=self.VV,
                                        HH=self.HH,
                                        VVdB=dB(self.VV),
                                        HHdB=dB(self.HH))

            self.BRDF = EmissivityResult(
                array=np.array([BRDF(self.VV, self.iza, self.vza)[0], BRDF(self.HH, self.iza, self.vza)[0]]).flatten(),
                arraydB=np.array(
                    [dB(BRDF(self.VV, self.iza, self.vza))[0], dB(BRDF(self.HH, self.iza, self.vza))[0]]).flatten(),
                VV=BRDF(self.VV, self.iza, self.iza),
                HH=BRDF(self.HH, self.iza, self.iza),
                VVdB=dB(BRDF(self.VV, self.iza, self.iza)),
                HHdB=dB(BRDF(self.HH, self.iza, self.iza)))

            self.BRF = EmissivityResult(array=np.array([BRF(self.BRDF.VV)[0], BRF(self.BRDF.HH)[0]]).flatten(),
                                        arraydB=np.array(
                                            [dB(BRF(self.BRDF.VV))[0], dB(BRF(self.BRDF.HH))[0]]).flatten(),
                                        VV=BRF(self.BRDF.VV),
                                        HH=BRF(self.BRDF.HH),
                                        VVdB=dB(BRF(self.BRDF.VV)),
                                        HHdB=dB(BRF(self.BRDF.HH)))

        def __spectrm(self, n_spec, nr, wvnb):

            wn = np.zeros([n_spec, nr])

            if self.corrfunc is 'exponential':  # exponential
                for n in srange(n_spec):
                    wn[n, :] = (n + 1) * self.kl ** 2 / ((n + 1) ** 2 + (wvnb * self.corrlen) ** 2) ** 1.5

            elif self.corrfunc is 'gaussian':  # gaussian
                for n in srange(n_spec):
                    wn[n, :] = 0.5 * self.kl ** 2 / (n + 1) * np.exp(-(wvnb * self.corrlen) ** 2 / (4 * (n + 1)))

            elif self.corrfunc is 'mixed':
                gauss = np.zeros([n_spec, nr])
                exp = np.zeros([n_spec, nr])

                for n in srange(n_spec):
                    gauss[n, :] = 0.5 * self.kl ** 2 / (n + 1) * np.exp(-(wvnb * self.corrlen) ** 2 / (4 * (n + 1)))

                for n in srange(n_spec):
                    exp[n, :] = (n + 1) * self.kl ** 2 / ((n + 1) ** 2 + (wvnb * self.corrlen) ** 2) ** 1.5

                wn = gauss / exp

            else:
                raise ValueError(
                    "Corrfunc must be 'exponential' or 'gaussian'. The actual value of corrfunc is: {}".format(
                        self.corrfunc))

            return wn

        def emsv_integralfunc(self, x, y):
            error = 1.0e3

            sqs = np.sqrt(self.diel_constant - np.sin(x) ** 2)
            rc = (self.rv - self.rh) / 2
            tv = 1 + self.rv
            th = 1 + self.rh

            # -- calc coefficients for surface correlation spectra
            wvnb = self.k * np.sqrt(
                np.sin(self.iza) ** 2 - 2 * np.sin(self.iza) * np.sin(x) * np.cos(y) + np.sin(x) ** 2)

            try:
                nr = len(x)

            except (IndexError, TypeError):
                nr = 1

            # -- calculate number of spectral components needed
            n_spec = 1
            while error > 1.0e-3:
                n_spec = n_spec + 1
                #   error = (ks2 *(cs + css)**2 )**n_spec / factorial(n_spec)
                # ---- in this case we will use the smallest ths to determine the number of
                # spectral components to use.  It might be more than needed for other angles
                # but this is fine.  This option is used to simplify calculations.
                error = (self.ks ** 2 * (np.cos(self.iza) + np.cos(x)) ** 2) ** n_spec / factorial(n_spec)
                error = np.min(error)
            # -- calculate expressions for the surface spectra
            wn = self.__spectrm(n_spec, nr, wvnb)

            # -- calculate fpq!

            ff = 2 * (np.sin(self.iza) * np.sin(x) - (1 + np.cos(self.iza) * np.cos(x)) * np.cos(y)) / (
                    np.cos(self.iza) + np.cos(x))

            fvv = self.rv * ff
            fhh = -self.rh * ff

            fvh = -2 * rc * np.sin(y)
            fhv = 2 * rc * np.sin(y)

            # -- calculate Fpq and Fpqs -----
            fhv = np.sin(self.iza) * (np.sin(x) - np.sin(self.iza) * np.cos(y)) / (np.cos(self.iza) ** 2 * np.cos(x))
            T = (self.sq * (np.cos(self.iza) + self.sq) + np.cos(self.iza) * (
                    self.diel_constant * np.cos(self.iza) + self.sq)) / (
                        self.diel_constant * np.cos(self.iza) * (np.cos(self.iza) + self.sq) + self.sq * (
                        self.diel_constant * np.cos(self.iza) + self.sq))
            cm2 = np.cos(x) * self.sq / np.cos(self.iza) / sqs - 1
            ex = np.exp(-self.ks ** 2 * np.cos(self.iza) * np.cos(x))
            de = 0.5 * np.exp(-self.ks ** 2 * (np.cos(self.iza) ** 2 + np.cos(x) ** 2))

            if self.pol == 'vv':
                Fvv = (self.diel_constant - 1) * np.sin(self.iza) ** 2 * tv ** 2 * fhv / self.diel_constant ** 2
                Fhv = (T * np.sin(self.iza) * np.sin(self.iza) - 1. + np.cos(self.iza) / np.cos(x) + (
                        self.diel_constant * T * np.cos(self.iza) * np.cos(x) * (
                        self.diel_constant * T - np.sin(self.iza) * np.sin(self.iza)) - self.sq * self.sq) / (
                               T * self.diel_constant * self.sq * np.cos(x))) * (1 - rc * rc) * np.sin(y)

                Fvvs = -cm2 * self.sq * tv ** 2 * (
                        np.cos(y) - np.sin(self.iza) * np.sin(x)) / np.cos(
                    self.iza) ** 2 / self.diel_constant - cm2 * sqs * tv ** 2 * np.cos(y) / self.diel_constant - (
                               np.cos(x) * self.sq / np.cos(self.iza) / sqs / self.diel_constant - 1) * np.sin(
                    x) * tv ** 2 * (
                               np.sin(self.iza) - np.sin(x) * np.cos(y)) / np.cos(self.iza)
                Fhvs = -(np.sin(x) * np.sin(x) / T - 1 + np.cos(x) / np.cos(self.iza) + (
                        np.cos(self.iza) * np.cos(x) * (
                        1 - np.sin(x) * np.sin(x) * T) - T * T * sqs * sqs) / (
                                 T * sqs * np.cos(self.iza))) * (1 - rc * rc) * np.sin(y)

                # -- calculate the bistatic field coefficients ---

                svv = np.zeros([n_spec, nr])
                for n in srange(n_spec):
                    Ivv = fvv * ex * (self.ks * (np.cos(self.iza) + np.cos(x))) ** (n + 1) + (
                            Fvv * (self.ks * np.cos(x)) ** (n + 1) + Fvvs * (self.ks * np.cos(self.iza)) ** (n + 1)) / 2
                    Ihv = fhv * ex * (self.ks * (np.cos(self.iza) + np.cos(x))) ** (n + 1) + (
                            Fhv * (self.ks * np.cos(x)) ** (n + 1) + Fhvs * (self.ks * np.cos(self.iza)) ** (n + 1)) / 2

                wnn = wn[n, :] / factorial(n + 1)
                vv = wnn * (abs(Ivv)) ** 2
                hv = wnn * (abs(Ihv)) ** 2
                svv[n, :] = (de * (vv + hv) * np.sin(x) * (1 / np.cos(self.iza))) / (4 * np.pi)

                ref = np.sum([svv])  # adding all n terms stores in different rows

            if self.pol == 'hh':
                Fhh = -(self.diel_constant - 1) * th ** 2 * fhv
                Fvh = (np.sin(self.iza) * np.sin(self.iza) / T - 1. + np.cos(self.iza) / np.cos(x) + (
                        np.cos(self.iza) * np.cos(x) * (
                        1 - np.sin(self.iza) * np.sin(self.iza) * T) - T * T * self.sq * self.sq) / (
                               T * self.sq * np.cos(x))) * (1 - rc * rc) * np.sin(y)

                Fhhs = cm2 * self.sq * th ** 2 * (
                        np.cos(y) - np.sin(self.iza) * np.sin(x)) / np.cos(
                    self.iza) ** 2 + cm2 * sqs * th ** 2 * np.cos(y) + cm2 * np.sin(x) * th ** 2 * (
                               np.sin(self.iza) - np.sin(x) * np.cos(y)) / np.cos(self.iza)
                Fvhs = -(T * np.sin(x) * np.sin(x) - 1 + np.cos(x) / np.cos(self.iza) + (
                        self.diel_constant * T * np.cos(self.iza) * np.cos(x) * (
                        self.diel_constant * T - np.sin(x) * np.sin(x)) - sqs * sqs) / (
                                 T * self.diel_constant * sqs * np.cos(self.iza))) * (1 - rc * rc) * np.sin(y)

                shh = np.zeros([n_spec, nr])
                for n in srange(n_spec):
                    Ihh = fhh * ex * (self.ks * (np.cos(self.iza) + np.cos(x))) ** (n + 1) + (
                            Fhh * (self.ks * np.cos(x)) ** (n + 1) + Fhhs * (self.ks * np.cos(self.iza)) ** (n + 1)) / 2
                    Ivh = fvh * ex * (self.ks * (np.cos(self.iza) + np.cos(x))) ** (n + 1) + (
                            Fvh * (self.ks * np.cos(x)) ** (n + 1) + Fvhs * (self.ks * np.cos(self.iza)) ** (n + 1)) / 2

                wnn = wn[n, :] / factorial(n + 1)
                hh = wnn * (abs(Ihh)) ** 2
                vh = wnn * (abs(Ivh)) ** 2
                (2 * (3 + 4) * np.sin(5) * 1 / np.cos(6)) / (np.pi * 4)
                shh[n, :] = (de * (hh + vh) * np.sin(x) * (1 / np.cos(self.iza))) / (4 * np.pi)

                ref = np.sum([shh])

            return ref
