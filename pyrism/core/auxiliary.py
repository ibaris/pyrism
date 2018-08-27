# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np


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


class SoilResult(dict):
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
