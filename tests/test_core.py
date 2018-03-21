import pytest
from numpy import radians, allclose, array

from pyrism.core import Kernel


@pytest.mark.webtest
@pytest.mark.parametrize("izaDeg, vzaDeg, raaDeg, izaRad, vzaRad, raaRad", [
    (0, 0, 0, radians(0), radians(0), radians(0)),
    (35, 45, 50, radians(35), radians(45), radians(50))
])
class TestKernelAngle:
    def test_kernel_DEG_iza(self, izaDeg, vzaDeg, raaDeg, izaRad, vzaRad, raaRad):
        kernel = Kernel(izaDeg, vzaDeg, raaDeg)
        assert allclose(izaRad, kernel.iza[0], atol=1e-4)

    def test_kernel_DEG_vza(self, izaDeg, vzaDeg, raaDeg, izaRad, vzaRad, raaRad):
        kernel = Kernel(izaDeg, vzaDeg, raaDeg)
        assert allclose(vzaRad, kernel.vza[0], atol=1e-4)

    def test_kernel_DEG_raa(self, izaDeg, vzaDeg, raaDeg, izaRad, vzaRad, raaRad):
        kernel = Kernel(izaDeg, vzaDeg, raaDeg)
        assert allclose(raaRad, kernel.raa[0], atol=1e-4)

    def test_kernel_RAD_iza(self, izaRad, vzaRad, raaRad, izaDeg, vzaDeg, raaDeg):
        kernel = Kernel(izaRad, vzaRad, raaRad, angle_unit='RAD')
        assert allclose(izaDeg, kernel.izaDeg[0], atol=1e-4)

    def test_kernel_RAD_vza(self, izaRad, vzaRad, raaRad, izaDeg, vzaDeg, raaDeg):
        kernel = Kernel(izaRad, vzaRad, raaRad, angle_unit='RAD')
        assert allclose(vzaDeg, kernel.vzaDeg[0], atol=1e-4)

    def test_kernel_RAD_raa(self, izaRad, vzaRad, raaRad, izaDeg, vzaDeg, raaDeg):
        kernel = Kernel(izaRad, vzaRad, raaRad, angle_unit='RAD')
        assert allclose(raaDeg, kernel.raaDeg[0], atol=1e-4)

    def test_align(self, izaRad, vzaRad, raaRad, izaDeg, vzaDeg, raaDeg):
        kernel = Kernel(array([izaRad, 2, 3]), array([izaRad, 2]), array([izaRad]), angle_unit='RAD')
        assert len(array([izaRad, 2, 3])) == len(kernel.raaDeg)

    def test_align_except(self, izaRad, vzaRad, raaRad, izaDeg, vzaDeg, raaDeg):
        with pytest.raises(AssertionError):
            Kernel(array([izaRad, 2, 3]), array([izaRad, 2]), array([izaRad]), angle_unit='RAD', align=False)


@pytest.mark.webtest
@pytest.mark.parametrize("izaDeg, vzaDeg, raaDeg, distance_true, piza_true, pvza_true, overlap_true", [
    (0, 0, 0, 0, 0, 0, 1),
    (35, 45, 50, 0.76819342, 0.61086524, 0.78539816, 0.23459992)
])
class TestKernelsGeometricSparse:
    def test_piza_sparse(self, izaDeg, vzaDeg, raaDeg, distance_true, piza_true, pvza_true, overlap_true):
        kernel = Kernel(izaDeg, vzaDeg, raaDeg)
        piza = kernel.get_proj_angle(1, kernel.iza)
        assert allclose(piza_true, piza, atol=1e-4)

    def test_pvza_sparse(self, izaDeg, vzaDeg, raaDeg, distance_true, piza_true, pvza_true, overlap_true):
        kernel = Kernel(izaDeg, vzaDeg, raaDeg)
        pvza = kernel.get_proj_angle(1, kernel.vza)
        assert allclose(pvza_true, pvza, atol=1e-4)

    def test_distance_sparse(self, izaDeg, vzaDeg, raaDeg, distance_true, piza_true, pvza_true, overlap_true):
        kernel = Kernel(izaDeg, vzaDeg, raaDeg)
        piza = kernel.get_proj_angle(1, kernel.iza)
        pvza = kernel.get_proj_angle(1, kernel.vza)
        _, distance = kernel.get_distance_function(piza, pvza, kernel.phi)

        assert allclose(distance_true, distance, atol=1e-4)

    def test_overlap_sparse(self, izaDeg, vzaDeg, raaDeg, distance_true, piza_true, pvza_true, overlap_true):
        kernel = Kernel(izaDeg, vzaDeg, raaDeg)
        piza = kernel.get_proj_angle(1, kernel.iza)
        pvza = kernel.get_proj_angle(1, kernel.vza)
        overlap = kernel.get_overlap(2.0, piza, pvza, kernel.phi)

        assert allclose(overlap_true, overlap, atol=1e-4)


@pytest.mark.webtest
@pytest.mark.parametrize("izaDeg, vzaDeg, raaDeg, distance_true, piza_true, pvza_true, overlap_true", [
    (0, 0, 0, 0, 0, 0, 1),
    (35, 45, 50, 1.92048356, 1.0517779, 1.19028995, 0)
])
class TestKernelsGeometricDense:
    def test_piza_dense(self, izaDeg, vzaDeg, raaDeg, distance_true, piza_true, pvza_true, overlap_true):
        kernel = Kernel(izaDeg, vzaDeg, raaDeg)
        piza = kernel.get_proj_angle(2.5, kernel.iza)
        assert allclose(piza_true, piza, atol=1e-4)

    def test_pvza_dense(self, izaDeg, vzaDeg, raaDeg, distance_true, piza_true, pvza_true, overlap_true):
        kernel = Kernel(izaDeg, vzaDeg, raaDeg)
        pvza = kernel.get_proj_angle(2.5, kernel.vza)
        assert allclose(pvza_true, pvza, atol=1e-4)

    def test_distance_dense(self, izaDeg, vzaDeg, raaDeg, distance_true, piza_true, pvza_true, overlap_true):
        kernel = Kernel(izaDeg, vzaDeg, raaDeg)
        piza = kernel.get_proj_angle(2.5, kernel.iza)
        pvza = kernel.get_proj_angle(2.5, kernel.vza)
        _, distance = kernel.get_distance_function(piza, pvza, kernel.phi)

        assert allclose(distance_true, distance, atol=1e-4)

    def test_overlap_dense(self, izaDeg, vzaDeg, raaDeg, distance_true, piza_true, pvza_true, overlap_true):
        kernel = Kernel(izaDeg, vzaDeg, raaDeg)
        piza = kernel.get_proj_angle(2.5, kernel.iza)
        pvza = kernel.get_proj_angle(2.5, kernel.vza)
        overlap = kernel.get_overlap(2.0, piza, pvza, kernel.phi)

        assert allclose(overlap_true, overlap, atol=1e-4)
