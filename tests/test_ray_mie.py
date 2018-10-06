import pytest
from numpy import allclose, array

from pyrism import Rayleigh, Mie
import numpy as np


# ---- Test Rayleigh ----

# Test coefficients
@pytest.mark.webtest
@pytest.mark.parametrize("freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true", [
    (1.26, 1, 0.25 + 0.1j, 1 + 1j, 0.0019, 0.0501, 0.0520, 0.0028)
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

    def rest_rayleigh_length(self, freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true):
        # After that we specify the sensing geometry we want to simulate
        radius = np.arange(0, 0.0205, 0.0005)

        ray = Rayleigh(frequency=1.26, radius=radius, eps_p=(0.25 + 0.1j))

        assert len(ray.ks) == len(radius)
        assert len(ray.ka) == len(radius)
        assert len(ray.ke) == len(radius)
        assert len(ray.BSC) == len(radius)


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
        mat = Rayleigh.Phase(iza, vza, raa)
        mat.dblquad()

        p11_ = mat.p11
        p12_ = mat.p12

        p21_ = mat.p21
        p22_ = mat.p22

        assert allclose(p11_, p11, atol=1e-4)
        assert allclose(p12_, p12, atol=1e-4)

        assert allclose(p21_, p21, atol=1e-4)
        assert allclose(p22_, p22, atol=1e-4)

    def test_rayleighphase2(self, p11, p12, p21, p22, iza, vza, raa):
        iza = np.arange(10, 30, 1)  # Incidence zenith angle
        vza = 30  # Viewing zenith angle
        raa = 50  # Relative azimuth angle

        mat = Rayleigh.Phase(iza, vza, raa)
        mat.dblquad()

        p11_ = mat.p11
        p12_ = mat.p12

        p21_ = mat.p21
        p22_ = mat.p22

        assert len(p11_) == len(iza)
        assert len(p12_) == len(iza)

        assert len(p21_) == len(iza)
        assert len(p22_) == len(iza)

# ---- Test Mie ----

# Test Scattering Coefficients
@pytest.mark.webtest
@pytest.mark.parametrize("freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true", [
    (1.26, 1, 0.25 + 0.1j, 1 + 1j, 0.0017, 0.0480, 0.0498, 0.0026)
])
class TestScatteringMie:
    def test_mie(self, freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true):
        freq = 1.26
        a = 0.001
        eps_1 = 0.25 + 0.1j
        r = Mie(freq, a, eps_1)
        print (r)
        result = array([r.ks, r.ka, r.ke, r.BSC])
        true = array([0.0017, 0.0480, 0.0498, 0.0026])

        assert allclose(result[0], true[0], atol=1e-4)
        assert allclose(result[1], true[1], atol=1e-4)
        assert allclose(result[2], true[2], atol=1e-4)
        assert allclose(result[3], true[3], atol=1e-4)
