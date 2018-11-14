from __future__ import division

from collections import namedtuple

import numpy as np

from ...volume.library import get_data_one, get_data_two
from ...auxil import SoilResult

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

    @property
    def I(self):
        return self.I

    def __calc(self):
        self.I = self.sRef * (self.moisture * lib.soil.rsoil1 + (1 - self.moisture) * lib.soil.rsoil2)
        self.int = [self.l, self.I]
        self.int = np.asarray(self.int, dtype=np.float32)
        self.int = self.int.transpose()

        self.I = SoilResult(array=self.I)

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
