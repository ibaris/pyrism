import numpy as np
import pytest

from pyrism import LIDF, VolScatt

result_campbell = np.array([8.31370856e-01, 1.18374448e-01, 2.60866256e-02,
                            9.68442862e-03, 4.68681271e-03, 2.66412346e-03,
                            1.68943850e-03, 1.16093072e-03, 8.49048566e-04,
                            6.53091633e-04, 5.24062841e-04, 4.36148636e-04,
                            3.74878009e-04, 3.31739471e-04, 3.01550650e-04,
                            2.81097639e-04, 2.68401688e-04, 2.62316948e-04])


@pytest.mark.webtest
@pytest.mark.parametrize("a, b, lidf_verhoef, lidf_campbell", [
    (0, 0, np.zeros(18) + 0.05555556, result_campbell)
])
class TestLIDF:
    def test_verhoef(self, a, b, lidf_verhoef, lidf_campbell):
        lidf_rom = LIDF.verhoef(a, b, n_elements=18)
        assert np.allclose(lidf_rom, lidf_verhoef, atol=1e-4)

    def test_campbell(self, a, b, lidf_verhoef, lidf_campbell):
        lidf_rom = LIDF.campbell(a, n_elements=18)
        assert np.allclose(lidf_rom, lidf_campbell, atol=1e-4)


@pytest.mark.webtest
@pytest.mark.parametrize("iza, vza, raa, a, b, ks, ko, bf, Fs, Ft", [
    (50, 30, 50, 0, 0, 0.823188863879162, 0.6867716425258737, 0.5, 0.6217351737543753, 0.011166182392885724)
])
class TestVolScatVerhoef:
    def test_vol_verhoef(self, iza, vza, raa, a, b, ks, ko, bf, Fs, Ft):
        vol = VolScatt(iza, vza, raa)
        vol.coef(a=a, b=b, lidf_type='verhoef')
        res = (vol.kei[0], vol.kev[0], vol.bf, vol.Fs[0], vol.Ft[0])
        true = (ks, ko, bf, Fs, Ft)
        assert np.allclose(res, true, atol=1e-4)


@pytest.mark.webtest
@pytest.mark.parametrize("iza, vza, raa, a, ks, ko, bf, Fs, Ft", [
    (50, 30, 50, 0, 0.9948675435577432, 0.9940524814594064, 0.9891027449943284, 0.9915874551449064,
     7.491316140639147e-05)
])
class TestVolScatVerhoef:
    def test_vol_verhoef(self, iza, vza, raa, a, ks, ko, bf, Fs, Ft):
        vol = VolScatt(iza, vza, raa)
        vol.coef(a=a, lidf_type='campbell')
        res = (vol.kei[0], vol.kev[0], vol.bf, vol.Fs[0], vol.Ft[0])
        true = (ks, ko, bf, Fs, Ft)
        assert np.allclose(res, true, atol=1e-4)
