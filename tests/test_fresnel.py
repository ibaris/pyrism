import pyrism as pyr
import numpy as np


class TestFresnel:
    def test_rmatrix(self):
        iza = np.arange(0, 90, 1)  # Incidence zenith angle
        vza = 30  # Viewing zenith angle
        raa = 50  # Relative azimuth angle

        frequency = 1.26  # Frequency in GHz
        eps = [1, 2 + 4j, 3 + 0.3j, 1 + 0.1j]  # Dielectric constant of soil
        corrlength = 10  # Correlation length in cm
        sigma = 0.5  # RMS Height in cm

        fresnel = pyr.Fresnel(xza=iza, frequency=frequency, eps=eps, sigma=sigma)

        assert len(fresnel.I.VV) == len(iza)
        assert len(fresnel.I.HH) == len(iza)
        assert len(fresnel.I.VH) == len(iza)
        assert len(fresnel.I.HV) == len(iza)

        assert len(fresnel.VV) == len(iza)
        assert len(fresnel.HH) == len(iza)
        assert len(fresnel.VH) == len(iza)
        assert len(fresnel.HV) == len(iza)

    def test_ematrix(self):
        iza = np.arange(0, 90, 10)  # Incidence zenith angle
        vza = 30  # Viewing zenith angle
        raa = 50  # Relative azimuth angle

        frequency = 1.26  # Frequency in GHz
        eps = [1, 2 + 4j, 3 + 0.3j, 1 + 0.1j]  # Dielectric constant of soil
        corrlength = 10  # Correlation length in cm
        sigma = 0.5  # RMS Height in cm

        fresnel = pyr.Fresnel.Emissivity(xza=iza, frequency=frequency, eps=eps, sigma=sigma)

        assert len(fresnel.EMS.VV) == len(iza)
        assert len(fresnel.EMS.HH) == len(iza)
        assert len(fresnel.EMS.VH) == len(iza)
        assert len(fresnel.EMS.HV) == len(iza)

        assert len(fresnel.VV) == len(iza)
        assert len(fresnel.HH) == len(iza)
        assert len(fresnel.VH) == len(iza)
        assert len(fresnel.HV) == len(iza)
