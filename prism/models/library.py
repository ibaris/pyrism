# -*- coding: utf-8 -*-
from __future__ import division

import os
from collections import namedtuple

import numpy as np
import pkg_resources

path = 'prism/models'  # os.path.split(__file__)

# Create a header
Spectra = namedtuple('Spectra', 'p5 pd soil light')
P5S = namedtuple('P5S', 'KN Kab Kxc Kbr Kw Km')
PDS = namedtuple('PDS', 'KN Kab Kxc Kbr Kw Km Kan')
SoilS = namedtuple("SoilS", "rsoil1 rsoil2")
LightS = namedtuple("LightS", "es ed")


def get_data_one():
    """
    Load spectral information for PROSPECT 5B, PROSPECT D and SAIL Model in to
    a namedtuple. The data is stored in the directory "data".

    Returns
    -------
    spectral : namedtuple
        Named tuple with following attributes (lib.px with x = 5 for PROSPECT 5 and x = d for PROSPECT D):
            * KN : Specific absorption coefficient of leaf structure parameter.
            * Kab : Specific absorption coefficient of chlorophyll (a+b).
            * Kxc : Specific absorption coefficient of carotenoids.
            * Kbr : Specific absorption coefficient of brown pigments.
            * Kw : Specific absorption coefficient of water.
            * Km : Specific absorption coefficient of dry matter.
            * Kan : Specific absorption coefficient of anthocyanins.
            * rsoil1 : Spectral measurements of a dry soil.
            * rsoil2 : Spectral measurements of a wet soil.

    References
    ----------
    .. [FeGN17]  Féret, Gitelson, Noble & Jacqumoud (2017). PROSPECT-D: Towards modeling
        leaf optical properties through a complete lifecycle
        Remote Sensing of Environment.

    .. [Bare]  The specific absorption coefficient corresponding to brown pigment is
        provided by Frederic Baret (EMMAH, INRA Avignon, baret@avignon.inra.fr)
        and used with his autorization.
    """
    # PROSPECT-D
    path = 'data/prospect_d_spectra.txt'  # always use slash
    filepath = pkg_resources.resource_filename(__name__, path)
    _, KN, Kab, Kxc, Kan, Kbr, Kw, Km = np.loadtxt(filepath, unpack=True, dtype=np.float32)

    pds = PDS(KN, Kab, Kxc, Kbr, Kw, Km, Kan)

    # PROSPECT 5
    path = 'data/prospect5_spectra.txt'  # always use slash
    filepath = pkg_resources.resource_filename(__name__, path)
    KN, Kab, Kxc, Kbr, Kw, Km = np.loadtxt(filepath, unpack=True, dtype=np.float32)

    p5s = P5S(KN, Kab, Kxc, Kbr, Kw, Km)

    # SOIL
    path = 'data/soil_reflectance.txt'  # always use slash
    filepath = pkg_resources.resource_filename(__name__, path)
    rsoil1, rsoil2 = np.loadtxt(filepath, unpack=True, dtype=np.float32)

    soils = SoilS(rsoil1, rsoil2)

    # LIGHT
    path = 'data/light_spectra.txt'  # always use slash
    filepath = pkg_resources.resource_filename(__name__, path)
    es, ed = np.loadtxt(filepath, unpack=True, dtype=np.float32)

    lights = LightS(es, ed)

    spectra = Spectra(p5s, pds, soils, lights)

    return spectra


def get_data_two():
    """
    Load spectral information for PROSPECT 5B, PROSPECT D and SAIL Model in to
    a namedtuple. The data is stored in the directory "data".
    Returns
    -------
    spectral : namedtuple
        Named tuple with following attributes (lib.px with x = 5 for PROSPECT 5 and x = d for PROSPECT D):
            * KN : Specific absorption coefficient of leaf structure parameter.
            * Kab : Specific absorption coefficient of chlorophyll (a+b).
            * Kxc : Specific absorption coefficient of carotenoids.
            * Kbr : Specific absorption coefficient of brown pigments.
            * Kw : Specific absorption coefficient of water.
            * Km : Specific absorption coefficient of dry matter.
            * Kan : Specific absorption coefficient of anthocyanins.
            * rsoil1 : Spectral measurements of a dry soil.
            * rsoil2 : Spectral measurements of a wet soil.
    References
    ----------
    .. [FeGN17]  Féret, Gitelson, Noble & Jacqumoud (2017). PROSPECT-D: Towards modeling
        leaf optical properties through a complete lifecycle
        Remote Sensing of Environment.
    .. [Bare]  The specific absorption coefficient corresponding to brown pigment is
        provided by Frederic Baret (EMMAH, INRA Avignon, baret@avignon.inra.fr)
        and used with his autorization.
    """

    # PROSPECT-D
    _, KN, Kab, Kxc, Kan, Kbr, Kw, Km = np.loadtxt(os.path.join(path, 'data', 'prospect_d_spectra.txt'), unpack=True,
                                                   dtype=np.float32)

    pds = PDS(KN, Kab, Kxc, Kbr, Kw, Km, Kan)

    # PROSPECT 5
    KN, Kab, Kxc, Kbr, Kw, Km = np.loadtxt(os.path.join(path, 'data', 'prospect5_spectra.txt'), unpack=True,
                                           dtype=np.float32)

    p5s = P5S(KN, Kab, Kxc, Kbr, Kw, Km)

    # SOIL
    rsoil1, rsoil2 = np.loadtxt(os.path.join(path, 'data', 'soil_reflectance.txt'), unpack=True,
                                dtype=np.float32)

    soils = SoilS(rsoil1, rsoil2)

    # LIGHT
    es, ed = np.loadtxt(os.path.join(path, 'data', 'light_spectra.txt'), unpack=True,
                        dtype=np.float32)

    lights = LightS(es, ed)

    spectra = Spectra(p5s, pds, soils, lights)

    return spectra
