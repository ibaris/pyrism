import pytest
from numpy import allclose

from prism.core import (ReflectanceResult, EmissivityResult, SailResult, BRF, BSC, BRDF, dB, nan_to_num, sec,
                        find_kernel, cot, rad, align_all, minus_to_zero, load_param, linear)

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