import pytest
from numpy import allclose, array

from prism import Rayleigh, Mie


@pytest.mark.webtest
@pytest.mark.parametrize("freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true", [
    (1.26, 0.01, 0.25 + 0.1j, 1 + 1j, 0.0019, 0.0501, 0.0520, 0.0028)
])
class TestScatteringRay:
    def test_rayleigh(self, freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true):
        r = Rayleigh(freq, a, eps_1, eps_2)
        result = array([r.ks[0], r.ka[0], r.ke[0], r.s0[0]])
        true = array([ks_true, ka_true, ke_true, s0_true])

        assert allclose(result, true, atol=1e-4)


@pytest.mark.webtest
@pytest.mark.parametrize("freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true", [
    (1.26, 0.01, 0.25 + 0.1j, 1 + 1j, 0.0017, 0.0480, 0.0498, 0.0026)
])
class TestScatteringRay:
    def test_rayleigh(self, freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true):
        r = Mie(freq, a, eps_1, eps_2)
        result = array([r.ks[0], r.ka[0], r.ke[0], r.s0[0]])
        true = array([ks_true, ka_true, ke_true, s0_true])

        assert allclose(result, true, atol=1e-4)
