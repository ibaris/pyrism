import pytest
from numpy import allclose, array

from pyrism import Rayleigh, Mie


# ---- Test Rayleigh ----

# Test coefficients
@pytest.mark.webtest
@pytest.mark.parametrize("freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true", [
    (1.26, 0.010, 0.25 + 0.1j, 1 + 1j, 0.0019, 0.0501, 0.0520, 0.0028)
])
class TestScatteringRayCoef:
    """
    The true values were calculated with the codes from Ulaby from 'http://mrs.eecs.umich.edu/interactive_modules.html',
    module '8.12 - Mie and Rayleigh Scattering'.
    """

    def test_rayleigh(self, freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true):
        r = Rayleigh(frequency=freq, radius=a, eps_p=eps_1, eps_b=eps_2)

        assert allclose(round(r.ks, 4), ks_true, atol=1e-4)
        assert allclose(round(r.ka, 4), ka_true, atol=1e-4)
        assert allclose(round(r.ke, 4), ke_true, atol=1e-4)
        assert allclose(round(r.BSC, 4), s0_true, atol=1e-4)


# The "true" values are calculated with sympy.integrate. Look at sympy_integrate_results_rayleig_phase.py in the test
# directory
@pytest.mark.webtest
@pytest.mark.parametrize("p11, p12, p21, p22, "
                         "iza, vza, raa, ", [
                             (1.83259571459405, 2.35619449019235, 1.04719755119660, 3.14159265358979,
                              35, 30, 50)
                         ])
class TestScatteringRayPhase:
    """
    The true values were calculated with sympy integrals. See tests/data/sympy_integration/sympy_phase_quad.py
    """

    def test_rayleighphase(self, p11, p12, p21, p22, iza, vza, raa):
        mat = Rayleigh.pmatrix(iza, vza, raa, dblquad=True)

        p11_ = mat[0, 0]
        p12_ = mat[0, 1]

        p21_ = mat[1, 0]
        p22_ = mat[1, 1]

        assert allclose(p11_, p11, atol=1e-4)
        assert allclose(p12_, p12, atol=1e-4)

        assert allclose(p21_, p21, atol=1e-4)
        assert allclose(p22_, p22, atol=1e-4)

#
# # ---- Test Mie ----
#
# # Test Scattering Coefficients
# @pytest.mark.webtest
# @pytest.mark.parametrize("freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true", [
#     (1.26, 0.01, 0.25 + 0.1j, 1 + 1j, 0.0017, 0.0480, 0.0498, 0.0026)
# ])
# class TestScatteringMie:
#     def test_mie(self, freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true):
#         r = Mie(freq, a, eps_1, eps_2)
#         result = array([r.ks[0], r.ka[0], r.ke[0], r.s0[0]])
#         true = array([ks_true, ka_true, ke_true, s0_true])
#
#         assert allclose(result, true, atol=1e-4)
