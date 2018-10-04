from .tm_single import TMatrixSingle
from .tm_psd import TMatrixPSD
import numpy as np

PI = 3.14159265359


class TMatrix(TMatrixSingle, TMatrixPSD):
    def __init__(self, iza, vza, iaa, vaa, frequency, radius, eps, alpha=0.0, beta=0.0,
                 radius_type='REV', shape='SPH', orientation='AF', axis_ratio=1.0, orientation_pdf=None, n_alpha=5,
                 n_beta=10,
                 angle_unit='DEG', psd=None, max_radius=10, num_points=1024, angular_integration=False):

        if psd is None:
            TMatrixSingle.__init__(self, iza=iza, vza=vza, iaa=iaa, vaa=vaa, frequency=frequency, radius=radius,
                                   eps=eps,
                                   alpha=alpha, beta=beta, radius_type=radius_type, shape=shape,
                                   orientation=orientation, axis_ratio=axis_ratio, orientation_pdf=orientation_pdf,
                                   n_alpha=n_alpha,
                                   n_beta=n_beta, angle_unit=angle_unit)

        else:
            TMatrixPSD.__init__(self, iza=iza, vza=vza, iaa=iaa, vaa=vaa, frequency=frequency, radius=radius, eps=eps,
                                alpha=alpha, beta=beta, radius_type=radius_type, shape=shape,
                                orientation=orientation, axis_ratio=axis_ratio, orientation_pdf=orientation_pdf,
                                n_alpha=n_alpha,
                                n_beta=n_beta, angle_unit=angle_unit,

                                psd=psd, num_points=num_points, angular_integration=angular_integration,
                                max_radius=max_radius)

        self.k0 = (2 * PI) / self.wavelength
        self.a = self.k0 * radius

    def Mpq(self, N, pq):

        factor = complex(0, 2 * PI * N) / self.k0

        if pq is 'VV':
            return factor * self.S[0, 0]

        elif pq is 'HH':
            return factor * self.S[1, 1]

        elif pq is 'VH':
            return factor * self.S[0, 1]

        elif pq is 'HV':
            return factor * self.S[1, 0]
        else:
            raise AssertionError("pq must be VV, HH, VH or HV")

    def ke(self, N=3):
        kem = np.zeros((4, 4))

        k11 = -2 * self.Mpq(N, 'VV').real
        k12 = 0
        k13 = -self.Mpq(N, 'VH').real
        k14 = -self.Mpq(N, 'VV').imag

        k21 = 0
        k22 = -2 * self.Mpq(N, 'HH').real
        k23 = -self.Mpq(N, 'HV').real
        k24 = -self.Mpq(N, 'HV').imag

        k31 = -2 * self.Mpq(N, 'HV').real
        k32 = -2 * self.Mpq(N, 'VH').real
        k33 = -(self.Mpq(N, 'VV').real + self.Mpq(N, 'HH').real)
        k34 = (self.Mpq(N, 'VV').imag - self.Mpq(N, 'HH').imag)

        k41 = 2 * self.Mpq(N, 'HV').imag
        k42 = -2 * self.Mpq(N, 'VH').imag
        k43 = -(self.Mpq(N, 'VV').imag - self.Mpq(N, 'HH').imag)
        k44 = -(self.Mpq(N, 'VV').real + self.Mpq(N, 'HH').real)

        kem[0, 0] = k11
        kem[0, 1] = k12
        kem[0, 2] = k13
        kem[0, 3] = k14

        kem[1, 0] = k21
        kem[1, 1] = k22
        kem[1, 2] = k23
        kem[1, 3] = k24

        kem[2, 0] = k31
        kem[2, 1] = k32
        kem[2, 2] = k33
        kem[2, 3] = k34

        kem[3, 0] = k41
        kem[3, 1] = k42
        kem[3, 2] = k43
        kem[3, 3] = k44

        return kem
