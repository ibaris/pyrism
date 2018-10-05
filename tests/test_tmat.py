from __future__ import division
import numpy as np
import pyrism as pyr
from numpy import allclose, less

# some allowance for rounding errors etc
epsilon = 1e-7


class TestTMatrix():

    def test_single(self):
        """Test a single-orientation case
        """
        iza = 90
        vza = 90
        iaa = 0
        vaa = 180

        tm = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=2, frequency=4.612191661538, eps=complex(1.5, 0.5),
                         axis_ratio=1 / 0.6)

        S, Z = tm.S, tm.Z

        S_ref = np.array(
            [[complex(3.89338755e-02, -2.43467777e-01),
              complex(-1.11474042e-24, -3.75103868e-24)],
             [complex(1.11461702e-24, 3.75030914e-24),
              complex(-8.38637654e-02, 3.10409912e-01)]])

        Z_ref = np.array(
            [[8.20899248e-02, -2.12975199e-02, -1.94051304e-24,
              2.43057373e-25],
             [-2.12975199e-02, 8.20899248e-02, 2.00801268e-25,
              -1.07794906e-24],
             [1.94055633e-24, -2.01190190e-25, -7.88399525e-02,
              8.33266362e-03],
             [2.43215306e-25, -1.07799010e-24, -8.33266362e-03,
              -7.88399525e-02]])

        assert allclose(S, S_ref)
        assert allclose(Z, Z_ref)

    def test_adaptive_orient(self):
        """Test an adaptive orientation averaging case
        """
        iza = 90
        vza = 90
        iaa = 0
        vaa = 180

        pdf = pyr.Orientation.gaussian(std=20)
        tm = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa,
                         radius=2, frequency=4.612191661538, eps=complex(1.5, 0.5), axis_ratio=1 / 0.6,
                         orientation_pdf=pdf, orientation='AA')

        S, Z = tm.S, tm.Z

        S_ref = np.array(
            [[complex(6.49005717e-02, -2.42488000e-01),
              complex(-6.12697676e-16, -4.10602248e-15)],
             [complex(-1.50048180e-14, -1.64195485e-15),
              complex(-9.54176591e-02, 2.84758322e-01)]])

        Z_ref = np.array(
            [[7.89677648e-02, -1.37631854e-02, -7.45412599e-15,
              -9.23979111e-20],
             [-1.37631854e-02, 7.82165256e-02, 5.61975938e-15,
              -1.32888054e-15],
             [8.68047418e-15, 3.52110917e-15, -7.73358177e-02,
              5.14571155e-03],
             [1.31977116e-19, -3.38136420e-15, -5.14571155e-03,
              -7.65845784e-02]])

        assert allclose(S, S_ref)
        assert allclose(Z, Z_ref)

    def test_fixed_orient(self):
        """Test a fixed-point orientation averaging case
        """
        iza = 90
        vza = 90
        iaa = 0
        vaa = 180

        pdf = pyr.Orientation.gaussian(std=20)
        tm = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa,
                         radius=2, frequency=4.612191661538, eps=complex(1.5, 0.5), axis_ratio=1 / 0.6,
                         orientation_pdf=pdf, orientation='AF')

        S, Z = tm.S, tm.Z

        S_ref = np.array(
            [[complex(6.49006090e-02, -2.42487917e-01),
              complex(1.20257317e-11, -5.23022168e-11)],
             [complex(6.21754594e-12, 2.95662844e-11),
              complex(-9.54177082e-02, 2.84758158e-01)]])

        Z_ref = np.array(
            [[7.89748116e-02, -1.37649947e-02, -1.58053610e-11,
              -4.56295798e-12],
             [-1.37649947e-02, 7.82237468e-02, -2.85105399e-11,
              -3.43475094e-12],
             [2.42108565e-11, -3.92054806e-11, -7.73426425e-02,
              5.14654926e-03],
             [4.56792369e-12, -3.77838854e-12, -5.14654926e-03,
              -7.65915776e-02]])

        assert allclose(S, S_ref)
        assert allclose(Z, Z_ref)

    def test_psd(self):
        """Test a case that integrates over a particle size distribution
        """
        iza = 90
        vza = 90
        iaa = 0
        vaa = 180

        PSD = pyr.PSD(r0=0.5, mu=4, n0=1e3, rmax=5)
        psd = PSD.gamma

        tm = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa,
                         radius=1, frequency=4.612191661538, eps=complex(1.5, 0.5), axis_ratio=1 / 0.6,
                         psd=psd, num_points=500, max_radius=5, angular_integration=False)

        S, Z = tm.S, tm.Z

        S_ref = np.array(
            [[complex(1.02521928e+00, 6.76066598e-01),
              complex(6.71933838e-24, 6.83819665e-24)],
             [complex(-6.71933678e-24, -6.83813546e-24),
              complex(-1.10464413e+00, -1.05571494e+00)]])

        Z_ref = np.array(
            [[7.20540295e-02, -1.54020475e-02, -9.96222107e-25,
              8.34246458e-26],
             [-1.54020475e-02, 7.20540295e-02, 1.23279391e-25,
              1.40049088e-25],
             [9.96224596e-25, -1.23291269e-25, -6.89739108e-02,
              1.38873290e-02],
             [8.34137617e-26, 1.40048866e-25, -1.38873290e-02,
              -6.89739108e-02]])

        assert allclose(S, S_ref)
        assert allclose(Z, Z_ref)

    def test_rayleigh(self):
        """Test match with Rayleigh scattering for small spheres
        """
        iza = 90
        vza = 90
        iaa = 0
        vaa = 180

        tm = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa,
                         radius=1, frequency=0.299792458, eps=complex(1.5, 0.5), axis_ratio=1)

        S = tm.S

        wl = 100.0

        k = 2 * np.pi / wl
        m = complex(1.5, 0.5)
        r = 1.0

        S_ray = k ** 2 * (m ** 2 - 1) / (m ** 2 + 2) * r

        assert allclose(S[0, 0], S_ray, atol=1e-3)
        assert allclose(S[1, 1], -S_ray, atol=1e-3)
        assert less(abs(S[0, 1]), 1e-25)
        assert less(abs(S[1, 0]), 1e-25)

    # todo: Not working because the values are wrong!
    # def test_optical_theorem(self):
    #     """Optical theorem: test that for a lossless particle, Csca=Cext
    #     """
    #     iza = 90
    #     vza = 90
    #     iaa = 0
    #     vaa = 0
    #
    #     tm = pyr.TMatrixSingle(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=4.0, frequency=4.612191661538,
    #                            eps=complex(1.5, 0.0),
    #                            axis_ratio=1.0 / 0.6)
    #
    #     ksVV, kaVV, keVV, omegaVV, ksHH, kaHH, keHH, omegaHH = tm.calc_xsec()
    #
    #     assert less(abs(1.0 - omegaHH), 1e-6)
    #     assert less(abs(1.0 - omegaVV), 1e-6)

    def test_asymmetry(self):
        """Test calculation of the asymmetry parameter
        """

        iza = 90
        vza = 90
        iaa = 0
        vaa = 0

        tm1 = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=4.0, frequency=4.612191661538,
                          eps=complex(1.5, 0.5),
                          axis_ratio=1.0)

        av1, ah1 = tm1.asx

        iza = 180
        vza = 180
        iaa = 0
        vaa = 0

        tm2 = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=4.0, frequency=4.612191661538,
                          eps=complex(1.5, 0.5),
                          axis_ratio=1.0)

        av2, ah2 = tm2.asx

        assert less(abs(1 - av1 / av2), 1e-6)
        assert less(abs(1 - ah1 / ah2), 1e-6)

        iza = 90
        vza = 90
        iaa = 0
        vaa = 0

        tm1 = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=4e-4, frequency=4.612191661538,
                          eps=complex(1.5, 0.5),
                          axis_ratio=1.0)

        av1, ah1 = tm1.asx
        # Is the asymmetry parameter zero for small particles?

        assert less(av1, 1e-8)
        assert less(ah1, 1e-8)

    def test_against_mie(self):
        """Test scattering parameters against Mie results
        """

        iza = 90
        vza = 90
        iaa = 0
        vaa = 180

        tm = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa,
                         radius=1, frequency=29.9792458, eps=complex(3, 0.5))

        ksVV, ksHH = tm.ksx
        keVV, keHH = tm.kex
        av1, ah1 = tm.asx

        # Reference values computed with the Mie code of Maetzler
        sca_xsect_ref = 4.4471684294079958
        ext_xsect_ref = 7.8419745883848435
        asym_ref = 0.76146646088675629

        assert less(abs(1 - ksHH / sca_xsect_ref), 1e-6)
        assert less(abs(1 - keHH / ext_xsect_ref), 1e-6)
        assert less(abs(1 - ah1 / asym_ref), 1e-6)

    # def test_integrated_x_sca(self):
    #     """Test Rayleigh scattering cross section integrated over sizes.
    #     """
    #
    #     iza = 90
    #     vza = 90
    #     iaa = 0
    #     vaa = 180
    #
    #     m = complex(3.0, 0.5)
    #     K = (m ** 2 - 1) / (m ** 2 + 2)
    #     N0 = 10
    #     Lambda = 1e4
    #
    #     PSD = pyr.PSD(ilambda=Lambda, n0=N0, rmax=0.002/2)
    #     psd = PSD.exponential
    #
    #     tm = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=1, max_radius=0.002/2,
    #                         frequency=1, eps=m, psd=psd, num_points=256,
    #                         angular_integration=True)
    #
    #     ksxV, ksxH = tm.ksx
    #     # This size-integrated scattering cross section has an analytical value.
    #     # Check that we can reproduce it.
    #     sca_xsect_ref = 480 * N0 * np.pi ** 5 * abs(K) ** 2 / Lambda ** 7
    #
    #     assert less(abs(1 - ksxH / sca_xsect_ref), 1e-3)
