import pytest
from numpy import allclose, array

from pyrism import Rayleigh, Mie


# ---- Test Rayleigh ----

# Test coefficients
@pytest.mark.webtest
@pytest.mark.parametrize("freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true", [
    (1.26, 0.01, 0.25 + 0.1j, 1 + 1j, 0.0019, 0.0501, 0.0520, 0.0028)
])
class TestScatteringRayCoef:
    def test_rayleigh(self, freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true):
        r = Rayleigh(frequency=freq, particle_size=a, diel_constant_p=eps_1, diel_constant_b=eps_2)

        # assert allclose(r.ks[0], ks_true, atol=1e-4)
        assert allclose(r.ka[0], ka_true, atol=1e-4)
        assert allclose(r.ke[0], ke_true, atol=1e-4)
        assert allclose(r.BSC[0], s0_true, atol=1e-4)


# Test Phase Integration
# After that we specify the sensing geometry we want to simulate
iza = 35  # Incidence zenith angle
vza = 30  # Viewing zenith angle
raa = 50  # Relative azimuth angle


# The "true" values are calculated with sympy.integrate. Look at sympy_integrate_results_rayleig_phase.py in the test
# directory
@pytest.mark.webtest
@pytest.mark.parametrize("p11, p12, p13, p14, "
                         "p21, p22, p23, p24, "
                         "p31, p32, p33, p34, "
                         "p41, p42, p43, p44", [
                             (1.83259571459405, 2.35619449019235, 0, 0,
                              1.04719755119660, 3.14159265358979, 0, 0,
                              0, 0, 0, 0,
                              0, 0, 0, 2.72069904635133)
                         ])
class TestScatteringRayPhase:
    def test_rayleighphase(self, p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34, p41, p42, p43, p44):
        mat = Rayleigh.phase_matrix(iza, vza, raa, integrate=True)

        p11_ = mat[0, 0]
        p12_ = mat[0, 1]
        p13_ = mat[0, 2]
        p14_ = mat[0, 3]

        p21_ = mat[1, 0]
        p22_ = mat[1, 1]
        p23_ = mat[1, 2]
        p24_ = mat[1, 3]

        p31_ = mat[2, 0]
        p32_ = mat[2, 1]
        p33_ = mat[2, 2]
        p34_ = mat[2, 3]

        p41_ = mat[3, 0]
        p42_ = mat[3, 1]
        p43_ = mat[3, 2]
        p44_ = mat[3, 3]

        assert allclose(p11_, p11, atol=1e-4)
        assert allclose(p12_, p12, atol=1e-4)
        assert allclose(p13_, p13, atol=1e-4)
        assert allclose(p14_, p14, atol=1e-4)

        assert allclose(p21_, p21, atol=1e-4)
        assert allclose(p22_, p22, atol=1e-4)
        assert allclose(p23_, p23, atol=1e-4)
        assert allclose(p24_, p24, atol=1e-4)

        assert allclose(p31_, p31, atol=1e-4)
        assert allclose(p32_, p32, atol=1e-4)
        assert allclose(p33_, p33, atol=1e-4)
        assert allclose(p34_, p34, atol=1e-4)

        assert allclose(p41_, p41, atol=1e-4)
        assert allclose(p42_, p42, atol=1e-4)
        assert allclose(p43_, p43, atol=1e-4)
        assert allclose(p44_, p44, atol=1e-4)


# ---- Test Mie ----

# Test Scattering Coefficients
@pytest.mark.webtest
@pytest.mark.parametrize("freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true", [
    (1.26, 0.01, 0.25 + 0.1j, 1 + 1j, 0.0017, 0.0480, 0.0498, 0.0026)
])
class TestScatteringMie:
    def test_mie(self, freq, a, eps_1, eps_2, ks_true, ka_true, ke_true, s0_true):
        r = Mie(freq, a, eps_1, eps_2)
        result = array([r.ks[0], r.ka[0], r.ke[0], r.s0[0]])
        true = array([ks_true, ka_true, ke_true, s0_true])

        assert allclose(result, true, atol=1e-4)
