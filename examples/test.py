import numpy as np


def norm_wrap(func):
    def norm_return_wrapper(norm):
        if norm:
            return func(norm)[0:-1]
        else:
            return func(norm)

    return norm_return_wrapper


@norm_wrap
def S(norm):
    if norm:
        n = 10
    else:
        n = 9

    return np.zeros((n, 2, 2))














import pyrism as pyr

# ----------------------------------------------------------------------------------------------------------------------
# Vectorized
# ----------------------------------------------------------------------------------------------------------------------
radius = np.arange(0.01, 0.05, 0.01).astype(np.double)
radius_type = 1
wavelength = np.arange(0.2, 0.6, 0.1).astype(np.double)
eps_real = np.arange(1, 5, 1).astype(np.double)
eps_imag = np.arange(1, 5, 1).astype(np.double)
axis_ratio = np.arange(1.46, 1.49, 0.01).astype(np.double)
shape = -1

izaDeg = np.arange(35, 39, 1).astype(np.double)
vzaDeg = np.arange(25, 29, 1).astype(np.double)
iaaDeg = np.arange(100, 104, 1).astype(np.double)
vaaDeg = np.arange(50, 54, 1).astype(np.double)
alphaDeg = np.zeros_like(izaDeg)
betaDeg = np.zeros_like(izaDeg)

n_alpha = 5
n_beta = 10



import numpy as np
from pyrism import TMatrix

izaDeg = np.linspace(10, 30, 5)
vzaDeg = 30
iaaDeg = 150
vaaDeg = 100
radius = 0.03
eps = (10 + 5j)

tmat = TMatrix(iza=izaDeg, vza=vzaDeg, iaa=iaaDeg, vaa=vaaDeg, frequency=1.26, radius=radius, eps=eps, normalize=True)


tmat.N=[1,2]
# tmat = TMatrixSingle(iza=izaDeg, vza=vzaDeg, iaa=iaaDeg, vaa=vaaDeg, frequency=1.26, radius=radius, eps=eps)

S = tmat.S
Z = tmat.Z

qs = tmat.QS
qe = tmat.QE
omega = qs.base / qe.base

qas = tmat.QAS

import numpy as np

iza = 0.25
vza = 0.10
Rv0 = (0.4 + 0.9j)
eps = (4 + 2j)

sin_iza = np.sin(iza)
sin_vza = np.sin(vza)
cos_iza = np.cos(iza)
Rv02 = pow(Rv0, 2)

8 * Rv02 * sin_vza * (cos_iza + np.sqrt(eps - pow(sin_iza, 2))) / (cos_iza * np.sqrt(eps - pow(sin_iza, 2)))
