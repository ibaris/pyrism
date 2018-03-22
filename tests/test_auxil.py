import numpy as np
import pytest

from pyrism.core import (ReflectanceResult, EmissivityResult, SailResult, BRF, BSC, BRDF, dB, sec,
                         cot, linear, load_param)


class TestResultClass:
    def reflectance(self):
        test = ReflectanceResult(a=1, b=2)
        assert test.a == 1
        assert test.b == 2

    def emissivity(self):
        test = EmissivityResult(a=1, b=2)
        assert test.a == 1
        assert test.b == 2

    def sail(self):
        test = SailResult(a=1, b=2)
        assert test.a == 1
        assert test.b == 2

@pytest.mark.webtest
@pytest.mark.parametrize("iza, vza, raa, ref", [
    (35,30,50,0.01)
])
class TestRF:
    def test_BRF(self, iza, vza, raa, ref):
        test = BRF(ref)
        np.allclose(test, np.pi * ref)

    def test_BSC(self, iza, vza, raa, ref):
        test = BSC(ref, iza, vza)
        np.allclose(test, ref * np.cos(iza) * np.cos(vza) * 4 * np.pi)

    def test_BSC_DEG(self, iza, vza, raa, ref):
        test = BSC(ref, iza, vza, angle_unit='DEG')
        np.allclose(test, ref * np.cos(np.radians(iza)) * np.cos(np.radians(vza)) * (4 * np.pi))

    def test_BSC_error(self, iza, vza, raa, ref):
        with pytest.raises(ValueError):
            test = BSC(ref, iza, vza, angle_unit='xxx')

    def test_BRDF(self, iza, vza, raa, ref):
        test = BRDF(ref, iza, vza)
        np.allclose(test, ref / (np.cos(iza) * np.cos(vza) * (4 * np.pi)))

    def test_BRDF_DEG(self, iza, vza, raa, ref):
        test = BRDF(ref, iza, vza, angle_unit='DEG')
        np.allclose(test, ref / (np.cos(np.radians(iza)) * np.cos(np.radians(vza)) * (4 * np.pi)))

    def test_BRDF_error(self, iza, vza, raa, ref):
        with pytest.raises(ValueError):
            test = BRDF(ref, iza, vza, angle_unit='xxx')

    def test_dB(self, iza, vza, raa, ref):
        test = dB(ref)
        np.allclose(test, 10 * np.log10(ref))

    def test_linear(self, iza, vza, raa, ref):
        test1 = dB(ref)
        test2 = linear(ref)
        np.allclose(test1, test2)

class TestAuxil:
    def test_sec(self):
        test = sec(35)
        assert test == 1 / np.cos(35)

    def test_cot(self):
        test = cot(35)
        assert test == 1 / np.tan(35)


class TestParam:
    def test_W1(self):
        param = load_param()
        assert param.W1.hs == 0.3
        assert param.W2.hs == 0.55
        assert param.W3.hs == 0.60
