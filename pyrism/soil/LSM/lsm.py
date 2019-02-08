from __future__ import division

from collections import namedtuple

import numpy as np
from respy import Conversion
from pyrism.volume.library import get_data_one, get_data_two
from pyrism.auxil import OPTICAL_RANGE, store_ASTER, store_L8, OpticalResult, Satellite
from respy import Quantity

try:
    lib = get_data_two()
except IOError:
    lib = get_data_one()


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
    rho_soil : int or float
        Surface (Lambertian) reflectance in optical wavelength.
    mv : int or float
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

    def __init__(self, rho_soil, mv, vza=None, angle_unit='DEG'):

        if angle_unit is "rad":
            angle_unit = "RAD"
        elif angle_unit is "deg":
            angle_unit = "DEG"

        self.angle_unit = angle_unit

        self.EM = OPTICAL_RANGE
        self.wavelength = self.EM.wavelength
        self.frequency = self.EM.frequency
        self.wavenumber = self.EM.wavenumber

        self.rho_soil = Quantity(rho_soil, name="Soil Brightness Factor")

        self.mv = Quantity(mv, name="Soil Moisture (Vol. %)")

        result = self.rho_soil * (self.mv * lib.soil.rsoil1 + (1 - self.mv) * lib.soil.rsoil2)
        self.__conversion = Conversion(result, vza=vza, value_unit='BRDF', angle_unit=angle_unit)

        self.__I = self.__conversion.I
        self.__BRF = self.__conversion.BRF
        self.__BSC = self.__conversion.BSC

        array = np.array([self.wavelength, self.I])
        self.array = array.transpose()
        self.__store()

    def __repr__(self):
        return self.I.create_arraystr('LSM')

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

    def __store(self):

        sat_I = Satellite(self.I, name='Intensity')
        sat_BRF = Satellite(self.BRF, name='Bidirectional Reflectance Factor')
        sat_BSC = Satellite(self.BSC, name='Backscattering Coefficient')

        self.__L8 = OpticalResult(I=sat_I.L8, BRF=sat_BRF.L8, BSC=sat_BSC.L8)
        self.__ASTER = OpticalResult(I=sat_I.ASTER, BRF=sat_BRF.ASTER, BSC=sat_BSC.ASTER)
