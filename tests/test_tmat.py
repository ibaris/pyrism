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
                         axis_ratio=1 / 0.6, radius_unit='cm')

        S, Z = tm.SZ

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

    # def test_adaptive_orient(self):
    #     """Test an adaptive orientation averaging case
    #     """
    #     iza = 90
    #     vza = 90
    #     iaa = 0
    #     vaa = 180
    #
    #     pdf = pyr.Orientation.gaussian(std=20)
    #     tm = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa,
    #                      radius=2, frequency=4.612191661538, eps=complex(1.5, 0.5), axis_ratio=1 / 0.6,
    #                      orientation_pdf=pdf, orientation='AA', radius_unit='cm')
    #
    #     S, Z = tm.SZ
    #
    #     S_ref = np.array(
    #         [[complex(6.49005717e-02, -2.42488000e-01),
    #           complex(-6.12697676e-16, -4.10602248e-15)],
    #          [complex(-1.50048180e-14, -1.64195485e-15),
    #           complex(-9.54176591e-02, 2.84758322e-01)]])
    #
    #     Z_ref = np.array(
    #         [[7.89677648e-02, -1.37631854e-02, -7.45412599e-15,
    #           -9.23979111e-20],
    #          [-1.37631854e-02, 7.82165256e-02, 5.61975938e-15,
    #           -1.32888054e-15],
    #          [8.68047418e-15, 3.52110917e-15, -7.73358177e-02,
    #           5.14571155e-03],
    #          [1.31977116e-19, -3.38136420e-15, -5.14571155e-03,
    #           -7.65845784e-02]])
    #
    #     assert allclose(S, S_ref)
    #     assert allclose(Z, Z_ref)

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
                         orientation_pdf=pdf, orientation='AF', radius_unit='cm')

        S, Z = tm.SZ

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

    def test_rayleigh(self):
        """Test match with Rayleigh scattering for small spheres
        """
        iza = 90
        vza = 90
        iaa = 0
        vaa = 180

        tm = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa,
                         radius=1, frequency=0.299792458, eps=complex(1.5, 0.5), axis_ratio=1, radius_unit='cm')

        S = tm.S

        wl = 100.0

        k = 2 * np.pi / wl
        m = complex(1.5, 0.5)
        r = 1.0

        S_ray = k ** 2 * (m ** 2 - 1) / (m ** 2 + 2) * r

        assert allclose(S[0, 0, 0], S_ray, atol=1e-3)
        assert allclose(S[0, 1, 1], -S_ray, atol=1e-3)
        assert less(abs(S[0, 0, 1]), 1e-25)
        assert less(abs(S[0, 1, 0]), 1e-25)

    def test_optical_theorem(self):
        """Optical theorem: test that for a lossless particle, Csca=Cext
        """
        iza = 90
        vza = 90
        iaa = 0
        vaa = 0

        tm = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=4.0, frequency=4.612191661538,
                         eps=complex(1.5, 0.0), axis_ratio=1.0 / 0.6, radius_unit='cm')

        omega = tm.omega

        assert less(abs(1.0 - omega[0, 0]), 1e-6)
        assert less(abs(1.0 - omega[0, 1]), 1e-6)

    def test_asymmetry(self):
        """Test calculation of the asymmetry parameter
        """

        iza = 90
        vza = 90
        iaa = 0
        vaa = 0

        tm1 = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=4.0, frequency=4.612191661538,
                          eps=complex(1.5, 0.5),
                          axis_ratio=1.0, radius_unit='cm')

        QAS1 = tm1.QAS

        iza = 180
        vza = 180
        iaa = 0
        vaa = 0

        tm2 = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=4.0, frequency=4.612191661538,
                          eps=complex(1.5, 0.5), axis_ratio=1.0, radius_unit='cm')

        QAS2 = tm2.QAS

        assert less(abs(1 - QAS1[0, 0] / QAS2[0, 0]), epsilon)
        assert less(abs(1 - QAS1[0, 1] / QAS2[0, 1]), epsilon)

        iza = 90
        vza = 90
        iaa = 0
        vaa = 0

        tm3 = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=4e-4, frequency=4.612191661538,
                          eps=complex(1.5, 0.5), axis_ratio=1.0, radius_unit='cm')

        QAS3 = tm3.QAS
        # Is the asymmetry parameter zero for small particles?

        assert less(QAS3[0, 0], epsilon)
        assert less(QAS3[0, 1], epsilon)

    def test_against_mie(self):
        """Test scattering parameters against Mie results
        """
        iza = 90
        vza = 90
        iaa = 0
        vaa = 180

        tm = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=1, frequency=29.9792458, eps=complex(3, 0.5),
                         radius_unit='cm')

        Qs = tm.QS
        Qe = tm.QE
        Qas = tm.QAS

        # Reference values computed with the Mie code of Maetzler
        sca_xsect_ref = 4.4471684294079958
        ext_xsect_ref = 7.8419745883848435
        asym_ref = 0.76146646088675629

        assert less(abs(1 - Qs[0,0] / sca_xsect_ref), 1e-6)
        assert less(abs(1 - Qe[0,0] / ext_xsect_ref), 1e-6)
        assert less(abs(1 - Qas[0,0] / asym_ref), 1e-6)

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
    #     PSD = pyr.PSD(ilambda=Lambda, n0=N0, rmax=0.002 / 2)
    #     psd = PSD.exponential
    #
    #     tm = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=1, max_radius=0.002 / 2,
    #                      frequency=29.9792458, eps=m, psd=psd, num_points=256,
    #                      angular_integration=True)
    #
    #     ksxV, ksxH = tm.ksx
    #     # This size-integrated scattering cross section has an analytical value.
    #     # Check that we can reproduce it.
    #     sca_xsect_ref = 480 * N0 * np.pi ** 5 * abs(K) ** 2 / Lambda ** 7
    #
    #     assert less(abs(1 - ksxH / sca_xsect_ref), 1e-3)


class TestNormalize:
    def norma(self):
        iza = 90
        vza = 90
        iaa = 0
        vaa = 180

        tm =    pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=.02, frequency=4.612191661538,
                            eps=complex(1.5, 0.5), axis_ratio=1 / 0.6, normalize=True)

        tm_ref = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=.02, frequency=4.612191661538,
                             eps=complex(1.5, 0.5), axis_ratio=1 / 0.6, normalize=False)

        S, Z = tm.SZ
        S_ref, Z_ref = tm_ref.SZ

        assert np.allclose(Z + tm.Znorm, Z_ref)
        assert np.allclose(S + tm.Snorm, S_ref)


class TestOriantation:
    def testuniform(self):
        x = 0.5
        ref = 7.615338836939474e-05
        pdf = pyr.Orientation.uniform()
        pdf = pdf(x)

        assert (allclose(pdf, ref))


# class TestTMATCLASS:
#     def test_single(self):
#         iza = 90
#         vza = 90
#         iaa = 0
#         vaa = 180
#
#         tm = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=.2, frequency=4.612191661538, eps=complex(1.5, 0.5),
#                          axis_ratio=1 / 0.6, normalize=False)
#
#         tm_ref = pyr.scattering.tmat.TMatrixSingle(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=.2,
#                                                    frequency=4.612191661538,
#                                                    eps=complex(1.5, 0.5),
#                                                    axis_ratio=1 / 0.6, normalize=False)
#
#         assert allclose(tm.ksx, tm_ref.ksx)
#         assert allclose(tm.kex, tm_ref.kex)
#         assert allclose(tm.asx, tm_ref.asx)
#         assert allclose(tm.dblquad, tm_ref.dblquad)
#         assert allclose(tm.ifunc_Z(0.25, 0.10, vza, vaa, 0., 0., tm.nmax, tm.wavelength, 0),
#                         tm_ref.ifunc_Z(0.25, 0.10, vza, vaa, 0., 0., tm.nmax, tm.wavelength, 0))
#         assert allclose(tm.ifunc_Z(0.25, 0.10, vza, vaa, 0., 0., tm.nmax, tm.wavelength, 1),
#                         tm_ref.ifunc_Z(0.25, 0.10, vza, vaa, 0., 0., tm.nmax, tm.wavelength, 1))
#
#     def test_single(self):
#         iza = 90
#         vza = 90
#         iaa = 0
#         vaa = 180
#         PSD = pyr.PSD(r0=0.5, mu=4, n0=1e3, rmax=5)
#         psd = PSD.gamma
#
#         tm = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa,
#                          radius=1, frequency=4.612191661538, eps=complex(1.5, 0.5), axis_ratio=1 / 0.6,
#                          psd=psd, num_points=100, max_radius=2, angular_integration=True,
#                          normalize=False)
#
#         tm_ref = pyr.scattering.tmat.TMatrixPSD(iza=iza, vza=vza, iaa=iaa, vaa=vaa,
#                                                 radius=1, frequency=4.612191661538, eps=complex(1.5, 0.5),
#                                                 axis_ratio=1 / 0.6,
#                                                 psd=psd, num_points=100, max_radius=2, angular_integration=True,
#                                                 normalize=False)
#
#         assert allclose(tm.ksx, tm_ref.ksx)
#         assert allclose(tm.kex, tm_ref.kex)
#         assert allclose(tm.asx, tm_ref.asx)
#         assert tm.dblquad == None


class TestXSEC:
    def test_coef(self):
        iza = 90
        vza = 90
        iaa = 0
        vaa = 180

        tm = pyr.TMatrix(iza=iza, vza=vza, iaa=iaa, vaa=vaa, radius=.02, frequency=4.612191661538, eps=complex(1.5, 0.5),
                         axis_ratio=1 / 0.6, normalize=False)

        ks = tm.ks
        ka = tm.ka
        kt = tm.kt
        ke = tm.ke

        assert allclose(ks[0, 0] + ka[0, 0], ke[0, 0, 0])
        assert allclose(ks[0, 1] + ka[0, 1], ke[0, 1, 1])

        assert allclose(ks[0, 0] + ka[0, 0] + kt[0, 0], 1)
        assert allclose(ks[0, 1] + ka[0, 1] + kt[0, 1], 1)
