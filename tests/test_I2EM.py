import pytest
from numpy import allclose

from pyrism import I2EM


#
# iza, vza, raa, frequency, diel_constant, corrlength, sigma, outVV, outHH = 10, 30, 50, 1.26, 6.9076590988636735 + 0.55947861142615318j, 30, 3, -7.3495, -6.0050
#
# import numpy as np
#
# iza = np.arange(10, 20, 1)

# The benchmark is the code from Ulaby.
@pytest.mark.webtest
@pytest.mark.parametrize("iza, vza, raa, frequency, diel_constant, corrlength, sigma, outVV, outHH", [
    (0, 30, 50, 1.26, 6.9076590988636735 + 0.55947861142615318j, 30, 3, -7.8974, -7.6297),
    (10, 30, 50, 1.26, 6.9076590988636735 + 0.55947861142615318j, 30, 3, -7.3495, -6.0050),
    (20, 30, 50, 1.26, 6.9076590988636735 + 0.55947861142615318j, 30, 3, -8.0978, -5.5328),
    (30, 30, 50, 1.26, 6.9076590988636735 + 0.55947861142615318j, 30, 3, -10.3902, -6.4485),
    (40, 30, 50, 1.26, 6.9076590988636735 + 0.55947861142615318j, 30, 3, -13.6212, -8.0656),
    (50, 30, 50, 1.26, 6.9076590988636735 + 0.55947861142615318j, 30, 3, -17.3226, -9.6842),
])
class TestI2EM:
    def test_i2emVV(self, iza, vza, raa, frequency, diel_constant, corrlength, sigma, outVV, outHH):
        eim = I2EM(iza, vza, raa, frequency=frequency, eps=diel_constant, corrlength=corrlength, sigma=sigma,
                   roughness_unit='cm')

        assert allclose(outVV, eim.BSC.VVdB, atol=1e-2)
        assert allclose(outHH, eim.BSC.HHdB, atol=1e-1)


# @pytest.mark.webtest
# @pytest.mark.parametrize("iza, vza, raa, frequency, diel_constant, corrlength, sigma, outVV, outHH", [
#     (0, 30, 50, 1.26, 6.9076590988636735 + 0.55947861142615318j, 30, 3, 0.8194, 0.8192),
#     (10, 30, 50, 1.26, 6.9076590988636735 + 0.55947861142615318j, 30, 3, 0.8217, 0.8133),
#     (20, 30, 50, 1.26, 6.9076590988636735 + 0.55947861142615318j, 30, 3, 0.8292, 0.7949),
#     (30, 30, 50, 1.26, 6.9076590988636735 + 0.55947861142615318j, 30, 3, 0.8446, 0.7620),
#     (40, 30, 50, 1.26, 6.9076590988636735 + 0.55947861142615318j, 30, 3, 0.8716, 0.7119),
#     (50, 30, 50, 1.26, 6.9076590988636735 + 0.55947861142615318j, 30, 3, 0.9116, 0.6426),
# ])
# class TestI2EMEMS:
#     def test_i2em_ems_VV(self, iza, vza, raa, frequency, diel_constant, corrlength, sigma, outVV, outHH):
#
#         eim = I2EM.Emissivity(iza, frequency=frequency, eps=diel_constant, corrlength=corrlength,
#                               sigma=sigma)
#
#         assert allclose(outVV, eim.EMS.V, atol=1e-1)
#         assert allclose(outHH, eim.EMS.H, atol=1e-1)
