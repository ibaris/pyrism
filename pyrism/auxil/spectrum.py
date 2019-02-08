# -*- coding: utf-8 -*-
"""
Created on 07.02.2019 by Ismail Baris
"""
from __future__ import division
from respy import Quantity, EM, Bands
from pyrism.auxil.auxiliary import OPTICAL_RANGE, OpticalResult
import numpy as np


class Satellite(object):
    def __init__(self, value, name=None):
        self.wavelength = OPTICAL_RANGE.wavelength

        if len(self.wavelength) != len(value):
            raise AssertionError(
                "Value must contain continuous reflectance values from from 400 until 2500 nm with a length of "
                "2101. The actual length of value is {0}".format(str(len(value))))

        self.value = value
        self.name = name if name is not None else b''

        array = np.array([self.wavelength, self.value])
        self.array = array.transpose()

        self.__band_names = ['Band 1', 'Band 2', 'Band 3', 'Band 4', 'Band 5', 'Band 6', 'Band 7', 'Band 8', 'Band 9']
        self.__abbrev = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']

        self.L8 = self.Landsat8()
        self.ASTER = self.Aster()

    def Landsat8(self):
        """ Store Landsat 8 bands.
        Store reflectance for Landsat 8 bands

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
        b2 = (452, 452 + 60)
        b3 = (533, 533 + 57)
        b4 = (636, 636 + 37)
        b5 = (851, 851 + 28)
        b6 = (1566, 1566 + 85)
        b7 = (2107, 2107 + 187)

        LRefB2 = self.array[(self.array[:, 0] >= b2[0]) & (self.array[:, 0] <= b2[1])]
        LRefB3 = self.array[(self.array[:, 0] >= b3[0]) & (self.array[:, 0] <= b3[1])]
        LRefB4 = self.array[(self.array[:, 0] >= b4[0]) & (self.array[:, 0] <= b4[1])]
        LRefB5 = self.array[(self.array[:, 0] >= b5[0]) & (self.array[:, 0] <= b5[1])]
        LRefB6 = self.array[(self.array[:, 0] >= b6[0]) & (self.array[:, 0] <= b6[1])]
        LRefB7 = self.array[(self.array[:, 0] >= b7[0]) & (self.array[:, 0] <= b7[1])]

        bands = np.array([LRefB2[:, 1].mean(), LRefB3[:, 1].mean(), LRefB4[:, 1].mean(), LRefB5[:, 1].mean(),
                          LRefB6[:, 1].mean(), LRefB7[:, 1].mean()])

        SENSOR = 'LANDSAT 8'
        BAND_NAMES = self.__band_names[1:-2]
        abbrev = self.__abbrev[1:-2]
        spectra = ['Blue', 'Green', 'Red', 'NIR', 'SWIR 1', 'SWIR 2']

        names = list()
        if self.name == b'':
            for i, item in enumerate(BAND_NAMES):
                names.append(SENSOR + ' ' + item + ' ' + '(' + spectra[i] + ')')

        else:
            for i, item in enumerate(BAND_NAMES):
                names.append(self.name + ' ' + SENSOR + ' ' + item + ' ' + '(' + spectra[i] + ')')

        L8 = OpticalResult()

        for i, item in enumerate(BAND_NAMES):
            L8[abbrev[i]] = Quantity(bands[i], name=names[i])

        return L8

    def Aster(self):
        """ Store Aster bands.
        Store reflectance for Aster bands

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

        b1 = (520, 600)
        b2 = (630, 690)
        b3 = (760, 860)
        b4 = (1600, 1700)
        b5 = (2145, 2185)
        b6 = (2185, 2225)
        b7 = (2235, 2285)
        b8 = (2295, 2365)
        b9 = (2360, 2430)

        ARefB1 = self.array[(self.array[:, 0] >= b1[0]) & (self.array[:, 0] <= b1[1])]
        ARefB2 = self.array[(self.array[:, 0] >= b2[0]) & (self.array[:, 0] <= b2[1])]
        ARefB3 = self.array[(self.array[:, 0] >= b3[0]) & (self.array[:, 0] <= b3[1])]
        ARefB4 = self.array[(self.array[:, 0] >= b4[0]) & (self.array[:, 0] <= b4[1])]
        ARefB5 = self.array[(self.array[:, 0] >= b5[0]) & (self.array[:, 0] <= b5[1])]
        ARefB6 = self.array[(self.array[:, 0] >= b6[0]) & (self.array[:, 0] <= b6[1])]
        ARefB7 = self.array[(self.array[:, 0] >= b7[0]) & (self.array[:, 0] <= b7[1])]
        ARefB8 = self.array[(self.array[:, 0] >= b8[0]) & (self.array[:, 0] <= b8[1])]
        ARefB9 = self.array[(self.array[:, 0] >= b9[0]) & (self.array[:, 0] <= b9[1])]

        bands = np.array([ARefB1[:, 1].mean(), ARefB2[:, 1].mean(), ARefB3[:, 1].mean(), ARefB4[:, 1].mean(),
                          ARefB5[:, 1].mean(), ARefB6[:, 1].mean(), ARefB7[:, 1].mean(), ARefB8[:, 1].mean(),
                          ARefB9[:, 1].mean()])

        spectra = ['Green', 'Red', 'NIR', 'SWIR 1', 'SWIR 2', 'SWIR 3', 'SWIR 4', 'SWIR 5', 'SWIR 6']

        SENSOR = 'ASTER'
        BAND_NAMES = self.__band_names
        abbrev = self.__abbrev

        names = list()
        if self.name == b'':
            for i, item in enumerate(BAND_NAMES):
                names.append(SENSOR + ' ' + item + ' ' + '(' + spectra[i] + ')')

        else:
            for i, item in enumerate(BAND_NAMES):
                names.append(self.name + ' ' + SENSOR + ' ' + item + ' ' + '(' + spectra[i] + ')')

        ASTER = OpticalResult()

        for i, item in enumerate(BAND_NAMES):
            ASTER[abbrev[i]] = Quantity(bands[i], name=names[i])

        return ASTER

    def ndvi(self, red=None, nir=None, satellite='L8'):
        if nir is None or red is None:
            if satellite is 'L8':
                red_ = self.L8.B4
                nir_ = self.L8.B5

            elif satellite is 'ASTER':
                red_ = self.ASTER.B2
                nir_ = self.ASTER.B3

            else:
                raise ValueError('Parameter red AND nir must be defined or the parameter satellite must '
                                 'be `L8` or `ASTER`. Parameter red and nir are None and satellite '
                                 'is {0}'.format(str(satellite)))

        else:
            red_ = red
            nir_ = nir

        ndvi = (nir_ - red_) / (nir_ + red_)

        if hasattr(ndvi, 'quantity'):
            if nir is None or red is None:
                ndvi.set_name('NDVI (' + satellite + ')')
                ndvi.set_constant(True)
            else:
                ndvi.set_name('NDVI')
                ndvi.set_constant(True)

        return ndvi

    def sr(self, red=None, nir=None, satellite='L8'):
        if nir is None or red is None:

            if satellite is 'L8':
                red_ = self.L8.B4
                nir_ = self.L8.B5

            elif satellite is 'ASTER':
                red_ = self.ASTER.B2
                nir_ = self.ASTER.B3

            else:
                raise ValueError('Parameter red AND nir must be defined or the parameter satellite must '
                                 'be `L8` or `ASTER`. Parameter red and nir are None and satellite '
                                 'is {0}'.format(str(satellite)))

        else:
            red_ = red
            nir_ = nir

        sr = nir_ / red_

        if hasattr(sr, 'quantity'):
            if nir is None or red is None:
                sr.set_name('NIR, RED Ratio (' + satellite + ')')
                sr.set_constant(True)
            else:
                sr.set_name('NIR, RED Ratio')
                sr.set_constant(True)

        return sr
