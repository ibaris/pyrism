# -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
import warnings
from datetime import datetime
from os.path import expanduser

import numpy as np
from respy import Bands, EM

EPSILON = sys.float_info.epsilon  # typical floating-point calculation error
try:
    import cPickle as pickle
except ImportError:
    import pickle

BANDS = Bands('nm')

OPTICAL_RANGE = EM(BANDS.get_region('OPTICS'), 'nm', 'THz')


def denormalized_decorator(func):
    """ Denormalize arrays with ndim 1 or 3
    A property decorator to deal with normalization in class instances.
    This decorator cuts the last element of the output if the class parameter `normalize` is set to True.

    Parameters
    ----------
    func : callable

    Returns
    -------
    callable
    """

    def denormalized(self):
        if self.normalized_flag:
            result = func(self)

            return result[0:-1]
        else:
            return func(self)

    return denormalized


def normalized_decorator(func):
    """ Normalize arrays with ndim 3
    A property decorator to deal with normalization in class instances.
    This decorator cuts the last element of the output and substract it with the other elements if the class
    parameter `normalize` is set to True.

    Parameters
    ----------
    func : callable

    Returns
    -------
    callable
    """

    def denormalized(self):
        if self.normalized_flag:
            result = func(self)
            return result[0:-1] - result[-1]
        else:
            return func(self)

    return denormalized


def get_version():
    version = dict()

    with open("pyrism/version.py") as fp:
        exec (fp.read(), version)

    return version['__version__']


class OpticalResult(dict):
    """ Represents the reflectance result.

    Returns
    -------
    All returns are attributes!
    BSC.U, BSC.VV, BSC.HH, BSC.VH, BSC.HV, BSC.array : array_like
        Radar backscattering values in [linear]. BSC.array contains the results as an array
        like array([BSC.U, BSC.VV, BSC.HH, BSC.VH, BSC.HV]).
    BSCdB.U, BSCdB.VV, BSCdB.HH, BSCdB.VH, BSCdB.HV, BSCdB.array : array_like
        Radar backscattering values in [linear]. BSC.array contains the results as an array
        like array([BSCdB.VV, BSCdB.HH, BSCdB.VH, BSCdB.HV]).
    I.U, I.VV, I.HH, I.VH, I.HV, I.array : array_like
        Intensity (BRDF) values. I.array contains the results as an array like array([I.U, I.VV, I.HH, I.VH, I.HV]).
    Bv.U, Bv.VV, Bv.HH, Bv.VH, Bv.HV, Bv.array : array_like
        Emissivity values (if available). Bv.array contains the results as an array
        like array([Bv.U, Bv.VV, Bv.HH, Bv.VH, Bv.HV]).

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class PyrismResult(dict):
    """ Represents the reflectance result.

    Returns
    -------
    All returns are attributes!
    BSC.U, BSC.VV, BSC.HH, BSC.VH, BSC.HV, BSC.array : array_like
        Radar backscattering values in [linear]. BSC.array contains the results as an array
        like array([BSC.U, BSC.VV, BSC.HH, BSC.VH, BSC.HV]).
    BSCdB.U, BSCdB.VV, BSCdB.HH, BSCdB.VH, BSCdB.HV, BSCdB.array : array_like
        Radar backscattering values in [linear]. BSC.array contains the results as an array
        like array([BSCdB.VV, BSCdB.HH, BSCdB.VH, BSCdB.HV]).
    I.U, I.VV, I.HH, I.VH, I.HV, I.array : array_like
        Intensity (BRDF) values. I.array contains the results as an array like array([I.U, I.VV, I.HH, I.VH, I.HV]).
    Bv.U, Bv.VV, Bv.HH, Bv.VH, Bv.HV, Bv.array : array_like
        Emissivity values (if available). Bv.array contains the results as an array
        like array([Bv.U, Bv.VV, Bv.HH, Bv.VH, Bv.HV]).

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __delattr__ = dict.__delitem__

    KEYLIST = ['array', 'BSC', 'BSCdB', 'Bv',
               'VV', 'HH', 'VH', 'HV', 'I', 'BRF']

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())

    def __setattr__(self, key, value):
        if key not in PyrismResultPol.KEYLIST:
            raise KeyError("{} is not a legal key of this PyrismResultPol".format(repr(key)))
        dict.__setitem__(self, key, value)


class Memorize(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class PyrismResultPol(dict):
    """ Represents the reflectance result.

    Returns
    -------
    All returns are attributes!
    BSC.U, BSC.VV, BSC.HH, BSC.VH, BSC.HV, BSC.array : array_like
        Radar backscattering values in [linear]. BSC.array contains the results as an array
        like array([BSC.U, BSC.VV, BSC.HH, BSC.VH, BSC.HV]).
    BSCdB.U, BSCdB.VV, BSCdB.HH, BSCdB.VH, BSCdB.HV, BSCdB.array : array_like
        Radar backscattering values in [linear]. BSC.array contains the results as an array
        like array([BSCdB.VV, BSCdB.HH, BSCdB.VH, BSCdB.HV]).
    I.U, I.VV, I.HH, I.VH, I.HV, I.array : array_like
        Intensity (BRDF) values. I.array contains the results as an array like array([I.U, I.VV, I.HH, I.VH, I.HV]).
    Bv.U, Bv.VV, Bv.HH, Bv.VH, Bv.HV, Bv.array : array_like
        Emissivity values (if available). Bv.array contains the results as an array
        like array([Bv.U, Bv.VV, Bv.HH, Bv.VH, Bv.HV]).

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __delattr__ = dict.__delitem__

    KEYLIST = ['array', 'U', 'BSC', 'BSCdB', 'Bv',
               'VV', 'HH', 'VH', 'HV']

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())

    def __setattr__(self, key, value):
        if key not in PyrismResultPol.KEYLIST:
            raise KeyError("{} is not a legal key of this PyrismResultPol".format(repr(key)))
        dict.__setitem__(self, key, value)


class SoilResult(dict):
    """ Represents the reflectance result.

    Returns
    -------
    All returns are attributes!
    BSC.ISO, BSC.VV, BSC.HH, BSC.ISOdB, BSC.VVdB, BSC.HHdB, BSC.array, BSC,arraydB : array_like
        Radar Backscatter values. BSC.array contains the results as an array like array([[BSC.VV], [BSC.HH]]).
    I.ISO, I.VV, I.HH, I.ISOdB, I.VVdB, I.HHdB, I.array, I,arraydB : array_like
        Intensity (BRDF) values. BRDF.array contains the results as an array like array([[BRDF.VV], [BRDF.HH]]).
    BRF.ISO, BRF.VV, BRF.HH, BRF.ISOdB, BRF.VVdB, BRF.HHdB, BRF.array, BRF,arraydB : array_like
        BRF reflectance values (polarization-dependent). BRF.array contains the results as an array like array([[BRF.VV], [BRF.HH]]).
    E.ISO, E.VV, E.HH, E.ISOdB, E.VVdB, E.HHdB, E.array, E,arraydB : array_like
        Emissivity values. E.array contains the results as an array like array([[E.VV], [E.HH]]).

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method. adar Backscatter values of multi scattering contribution of surface and volume
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class ReflectanceResult(dict):
    """ Represents the reflectance result.

    Returns
    -------
    All returns are attributes!
    BSC.ref, BSC.refdB : array_like
        Radar Backscatter values (polarization-independent).
    BSC.VV, BSC.HH, BSC.VVdB, BSC.HHdB, BSC.array, BSC,arraydB : array_like
        Radar Backscatter values (polarization-dependent). BSC.array contains the results as an array like array([[BSC.VV], [BSC.HH]]).
    BRDF.ref, BRDF.refdB : array_like
        BRDF reflectance values (polarization-independent).
    BRDF.VV, BRDF.HH, BRDF.VVdB, BRDF.HHdB, BRDF.array, BRDF,arraydB : array_like
        BRDF reflectance values (polarization-dependent). BRDF.array contains the results as an array like array([[BRDF.VV], [BRDF.HH]]).
    BRF.ref, BRF.refdB : array_like
        BRF reflectance values (polarization-independent).
    BRF.VV, BRF.HH, BRF.VVdB, BRF.HHdB, BRF.array, BRF,arraydB : array_like
        BRF reflectance values (polarization-dependent). BRF.array contains the results as an array like array([[BRF.VV], [BRF.HH]]).

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method. adar Backscatter values of multi scattering contribution of surface and volume

    The attribute 'ms' is the multi scattering contribution. This is only available if it is calculated. For detailed
    parametrisation one can use BSC.ms.sms or BSC.ms.smv for the multiple scattering contribution of surface or volume,
    respectively.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class SailResult(dict):
    """ Represents the sail result.

    Returns
    -------
    All returns are attributes!
    SDR.ref, SDR.refdB : array_like
        Directional reflectance factor.
    BHR.ref, BHR.refdB : array_like
        Bi-hemispherical reflectance factor.
    DHR.ref, DHR.refdB : array_like
        Directional-Hemispherical reflectance factor.
    HDR.ref, HDR.refdB : array_like
        Hemispherical-Directional reflectance factor.

    Note
    ----
    All returns have in addition the attributes `L8.Bx` and `ASTER.Bx`. L8 is the Landsat 8 average reflectance values
    for Bx band (B2 until B7). `ASTER` is the ASTER average reflectance for Bx band (B1 until B9).

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method. adar Backscatter values of multi scattering contribution of surface and volume

    The attribute 'ms' is the multi scattering contribution. This is only available if it is calculated. For detailed
    parametrisation one can use BSC.ms.sms or BSC.ms.smv for the multiple scattering contribution of surface or volume,
    respectively.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class EmissivityResult(dict):
    """ Represents the reflectance result.

    Returns
    -------
    All returns are attributes!
    EMN.VV, EMN.HH, EMN.VVdB, EMN.HHdB, EMN.array, EMN,arraydB : array_like : array_like
        Normalized Emission values (use EMS for emission values). Due to the several conversions in some other models
        this output format delivers the emission values divided through the
        sensing geometry times 4pi. This attribute is only for the I2EM.Emissivity class. If you want to calculate the
        emissivity of a scene, use this output from EMS.
        EMN.array contains the results as an array like array([[EMN.VV], [EMN.HH]]).
    EMS.VV, EMS.HH, EMS.VVdB, EMS.HHdB, EMN.array, EMN,arraydB : array_like
        Emission values. EMS.array contains the results as an array like array(][EMS.VV], [EMS.HH]]).

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method. adar Backscatter values of multi scattering contribution of surface and volume

    The attribute 'ms' is the multi scattering contribution. This is only available if it is calculated. For detailed
    parametrisation one can use BSC.ms.sms or BSC.ms.smv for the multiple scattering contribution of surface or volume,
    respectively.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def load_param():
    sensing = Memorize(freq=1.26,
                       iza=35,
                       vza=30,
                       raa=50)

    # W1
    W1 = Memorize(hs=0.3,
                  l=30,
                  S=0.8,
                  C=0.15,
                  mv=0.2,
                  pb=1.7,
                  p0=0.31831,
                  N=1,
                  Cab=20,
                  Cxc=3,
                  Cb=0.4,
                  Cw=0.0005,
                  Cm=0.0085,
                  LAI=1,
                  lza=45,
                  zmax=30,
                  bsp=15,
                  h=15,
                  r=3,
                  hb=1,
                  bspr=5,
                  mg=0.19,
                  T0=16,
                  a=0.02)

    W2 = Memorize(hs=0.55,
                  l=28,
                  S=0.8,
                  C=0.15,
                  mv=0.35,
                  pb=1.7,
                  p0=0.31831,
                  N=1.5,
                  Cab=35,
                  Cxc=5,
                  Cb=0.15,
                  Cw=0.003,
                  Cm=0.0055,
                  LAI=4,
                  lza=45,
                  zmax=30,
                  bsp=15,
                  h=15,
                  r=3,
                  hb=1,
                  bspr=5,
                  mg=0.28,
                  T0=20,
                  a=0.02)

    W3 = Memorize(hs=0.60,
                  l=25,
                  S=0.8,
                  C=0.15,
                  mv=0.45,
                  pb=1.7,
                  p0=0.31831,
                  N=2.2,
                  Cab=47,
                  Cxc=9,
                  Cb=0,
                  Cw=0.005,
                  Cm=0.002,
                  LAI=7,
                  lza=45,
                  zmax=30,
                  bsp=15,
                  h=15,
                  r=3,
                  hb=1,
                  bspr=5,
                  mg=0.51,
                  T0=24,
                  a=0.02)

    return Memorize(sensing=sensing,
                    W1=W1,
                    W2=W2,
                    W3=W3)


class Files(object):
    def __init__(self, path=None):
        if path is None:
            self.path = os.path.join(expanduser("~"), '.pyrism')
        else:
            self.path = os.path.join(path, '.pyrism')

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.refresh()

    @property
    def files(self):
        return self.__files

    def select_newest(self):
        diff = list()
        for i, files in enumerate(self.files):
            cont = files.split('-')
            date = datetime(year=int(cont[0]), month=int(cont[1]), day=int(cont[2]), hour=int(cont[3]),
                            minute=int(cont[4]), second=int(cont[5]))

            delta = datetime.now() - date
            minutes, seconds = divmod(delta.seconds + delta.days * 86400, 60)
            delta = minutes + (seconds * 0.016666666666667)

            diff.append(delta)

        val, idx = min((val, idx) for (idx, val) in enumerate(diff))

        return self.files[idx]

    def generate_fn(self, name='pyr-st'):
        time = datetime.now().isoformat()
        time = time.split('.')[0]
        time = time.replace(':', '-')
        time = time.replace('T', '-')

        return time + '-' + name

    def refresh(self):
        self.__files = os.listdir(self.path)

    def load_param(self, fn=None):
        """Load the scattering lookup tables.

        Load the scattering lookup tables saved with save_scatter_table.

        Args:
            fn: The name of the scattering table file.
        """

        if fn is None:
            fn = self.select_newest()
            data = pickle.load(open(os.path.join(self.path, fn)))
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


def ndvi(red, nir):
    return (nir - red) / (nir + red)


def sr(red, nir):
    return nir / red


def store_ASTER(value):
    """ Store Aster bands.
    Store reflectance for Aster bands

    Parameters
    ----------
    value : numpy.ndarray
        The reflectance value in a range from 400 nm - 2500 nm.

    Returns
    -------
    Aster bands B1 - B9: numpy.ndarray
        The mean values of Aster Bands from B1 - B9. See `Notes`.

    Note
    ----
    Landsat 8 Bands are in [nm]:
        * B1 = 520 - 600
        * B2 = 630 - 690
        * B3 = 760 - 860
        * B4 = 1600 - 1700
        * B5 = 2145 - 2185
        * B6 = 2185 - 2225
        * B7 = 2235 - 2285
        * B8 = 2295 - 2365
        * B9 = 2360 - 2430
    """

    wavelength = np.arange(400, 2501)

    if len(wavelength) != len(value):
        raise AssertionError(
            "Value must contain continuous reflectance values from from 400 until 2500 nm with a length of "
            "2101. The actual length of value is {0}".format(str(len(value))))

    value = np.array([wavelength, value])
    value = value.transpose()

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

    bands = np.array([ARefB1[:, 1].mean(), ARefB2[:, 1].mean(), ARefB3[:, 1].mean(), ARefB4[:, 1].mean(),
                      ARefB5[:, 1].mean(), ARefB6[:, 1].mean(), ARefB7[:, 1].mean(), ARefB8[:, 1].mean(),
                      ARefB9[:, 1].mean()])

    return bands


def store_L8(value):
    """ Store Landsat 8 bands.
    Store reflectance for Landsat 8 bands

    Parameters
    ----------
    value : numpy.ndarray
        The reflectance value in a range from 400 nm - 2500 nm.

    Returns
    -------
    Landsat bands B2 - B7: numpy.ndarray
        The mean values of Landsat 8 Bands from numpy.array([B2 - B7]). See `Notes`.

    Note
    ----
    Landsat 8 Bands are in [nm]:
        * B2 = 452 - 512
        * B3 = 533 - 590
        * B4 = 636 - 673
        * B5 = 851 - 879
        * B6 = 1566 - 1651
        * B7 = 2107 - 2294
    """
    wavelength = OPTICAL_RANGE.wavelength

    if len(wavelength) != len(value):
        raise AssertionError(
            "Value must contain continuous reflectance values from from 400 until 2500 nm with a length of "
            "2101. The actual length of value is {0}".format(str(len(value))))

    value = np.array([wavelength, value])
    value = value.transpose()

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

    bands = np.array([LRefB2[:, 1].mean(), LRefB3[:, 1].mean(), LRefB4[:, 1].mean(), LRefB5[:, 1].mean(),
                      LRefB6[:, 1].mean(), LRefB7[:, 1].mean()])

    return bands


def convert_L8_to_quantity(I, BRF, BSC):
    pass
