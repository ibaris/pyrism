import pytest
import numpy as np
from prism.core import (ReflectanceResult, EmissivityResult, SailResult, BRF, BSC, BRDF, dB, sec,
                        cot, linear)

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

    def test_BSC(self, iza, vza, raa, ref):
        test = BSC(ref, iza, vza)
        np.allclose(test, ref * np.cos(iza) * np.cos(vza) * 4 * np.pi)

    def test_BRDF(self, iza, vza, raa, ref):
        test = BRDF(ref, iza, vza)
        np.allclose(test, ref / (np.cos(iza) * np.cos(vza) * (4 * np.pi)))

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