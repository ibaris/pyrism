import pytest
from numpy import allclose

from prism import DielConstant


@pytest.mark.webtest
@pytest.mark.parametrize("freq, temp, sal, S, C, mv, rho_b, mg, water_true, water_sal_true, soil_true, veg_true", [
    (1.26, 20, 0.1, 0.6, 0.2, 0.05, 1.78, 0.2, 79.7592 + 5.5774j, 79.7232 + 5.8296j, 5.9635 + 0.4623j, 4.6792 + 1.5111j)
])
class TestDielConst:
    def test_water(self, freq, temp, sal, S, C, mv, rho_b, mg, water_true, water_sal_true, soil_true, veg_true):
        r = DielConstant.water(freq, temp)
        assert allclose(r, water_true, atol=1e-4)

    def test_water_sal(self, freq, temp, sal, S, C, mv, rho_b, mg, water_true, water_sal_true, soil_true, veg_true):
        r = DielConstant.saline_water(freq, temp, sal)
        assert allclose(r, water_sal_true, atol=1e-4)

    def test_soil(self, freq, temp, sal, S, C, mv, rho_b, mg, water_true, water_sal_true, soil_true, veg_true):
        r = DielConstant.soil(freq, temp, S, C, mv, rho_b)
        assert allclose(r, soil_true, atol=1e-4)

    def test_veg(self, freq, temp, sal, S, C, mv, rho_b, mg, water_true, water_sal_true, soil_true, veg_true):
        r = DielConstant.vegetation(freq, mg)
        assert allclose(r, veg_true, atol=1e-4)
