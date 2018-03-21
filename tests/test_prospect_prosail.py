import os
from distutils import dir_util

import pytest
from numpy import allclose, loadtxt
from pytest import fixture
from scipy.io import loadmat

from pyrism import PROSPECT, SAIL


@fixture
def datadir(tmpdir, request):
    """
    Fixture responsible for locating the test data directory and copying it
    into a temporary directory.
    Taken from  http://www.camillescott.org/2016/07/15/travis-pytest-scipyconf/
    """
    filename = request.module.__file__
    test_dir = os.path.dirname(filename)
    data_dir = os.path.join(test_dir, 'data')
    dir_util.copy_tree(data_dir, str(tmpdir))

    def getter(filename, as_str=True):
        filepath = tmpdir.join(filename)
        if as_str:
            return str(filepath)
        return filepath

    return getter


class TestPROSPECT:
    def test_reflectance_prospect5(self, datadir):
        # runs prospect and compares to online prospect run
        fname = datadir("prospect5_spectrum.txt")
        w, true_refl, true_trans = loadtxt(fname, unpack=True)

        prospect = PROSPECT(N=2.1, Cab=40, Cxc=10., Cbr=0.1, Cw=0.015, Cm=0.009, version='5')
        w, refl, trans = prospect.l, prospect.ks, prospect.kt

        assert allclose(true_refl, refl, atol=1e-4)

    def test_transmittance_prospect5(self, datadir):
        # runs prospect and compares to online prospect run
        fname = datadir("prospect5_spectrum.txt")
        w, true_refl, true_trans = loadtxt(fname, unpack=True)

        prospect = PROSPECT(N=2.1, Cab=40, Cxc=10., Cbr=0.1, Cw=0.015, Cm=0.009, version="5")
        w, refl, trans = prospect.l, prospect.ks, prospect.kt

        assert allclose(true_trans, trans, atol=1e-4)

    def test_reflectance_prospectd(self, datadir):
        fname = datadir("prospect_d_test.mat")
        refl_mtlab = loadmat(fname)['LRT'][:, 1]

        prospect = PROSPECT(N=1.2, Cab=30, Cxc=10., Cbr=0.0, Cw=0.015, Cm=0.009, Can=1, version="D")
        w, refl, trans = prospect.l, prospect.ks, prospect.kt

        assert allclose(refl_mtlab, refl, atol=1.e-4)

    def test_transmittance_prospectd(self, datadir):
        fname = datadir("prospect_d_test.mat")
        trans_mtlab = loadmat(fname)['LRT'][:, 2]

        prospect = PROSPECT(N=1.2, Cab=30, Cxc=10., Cbr=0.0, Cw=0.015, Cm=0.009, Can=1, version="D")
        w, refl, trans = prospect.l, prospect.ks, prospect.kt

        assert allclose(trans_mtlab, trans, atol=1.e-4)

    def test_raise_exception_version(self):
        with pytest.raises(ValueError):
            PROSPECT(N=2.1, Cab=40, Cxc=10., Cbr=0.1, Cw=0.015, Cm=0.009, Can=1, version="d")


class TestPROSAIL:
    def test_sdr_prosail5(self, datadir):
        fname = datadir("REFL_CAN.txt")
        w, resv, hdr, sdr, bhr, dhr = loadtxt(fname, unpack=True)

        prospect = PROSPECT(N=1.5, Cab=40, Cxc=8., Cbr=0.0, Cw=0.01, Cm=0.009, version="5")
        sail = SAIL(iza=30, vza=10, raa=0, ks=prospect.ks, kt=prospect.kt, lai=3, hotspot=0.01, soil_reflectance=1,
                    soil_moisture=1,
                    a=-0.35, b=-0.15, lidf_type='verhoef')

        assert allclose(sdr, sail.BRF.ref, atol=0.01)

    def test_hdr_prosail5(self, datadir):
        fname = datadir("REFL_CAN.txt")
        w, resv, hdr, sdr, bhr, dhr = loadtxt(fname, unpack=True)

        prospect = PROSPECT(N=1.5, Cab=40, Cxc=8., Cbr=0.0, Cw=0.01, Cm=0.009, version="5")
        sail = SAIL(iza=30, vza=10, raa=0, ks=prospect.ks, kt=prospect.kt, lai=3, hotspot=0.01, soil_reflectance=1,
                    soil_moisture=1,
                    a=-0.35, b=-0.15, lidf_type='verhoef')

        assert allclose(hdr, sail.HDR.ref, atol=0.01)

    def test_bhr_prosail5(self, datadir):
        fname = datadir("REFL_CAN.txt")
        w, resv, hdr, sdr, bhr, dhr = loadtxt(fname, unpack=True)

        prospect = PROSPECT(N=1.5, Cab=40, Cxc=8., Cbr=0.0, Cw=0.01, Cm=0.009, version="5")
        sail = SAIL(iza=30, vza=10, raa=0, ks=prospect.ks, kt=prospect.kt, lai=3, hotspot=0.01, soil_reflectance=1,
                    soil_moisture=1,
                    a=-0.35, b=-0.15, lidf_type='verhoef')

        assert allclose(bhr, sail.BHR.ref, atol=0.01)

    def test_dhr_prosail5(self, datadir):
        fname = datadir("REFL_CAN.txt")
        w, resv, hdr, sdr, bhr, dhr = loadtxt(fname, unpack=True)

        prospect = PROSPECT(N=1.5, Cab=40, Cxc=8., Cbr=0.0, Cw=0.01, Cm=0.009, version="5")
        sail = SAIL(iza=30, vza=10, raa=0, ks=prospect.ks, kt=prospect.kt, lai=3, hotspot=0.01, soil_reflectance=1,
                    soil_moisture=1,
                    a=-0.35, b=-0.15, lidf_type='verhoef')

        assert allclose(dhr, sail.DHR.ref, atol=0.01)
