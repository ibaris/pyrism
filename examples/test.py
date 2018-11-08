from pyrism.core.tma import (NMAX_VEC_WRAPPER, SZ_S_VEC_WRAPPER, SZ_AF_VEC_WRAPPER, NMAX_WRAPPER, SZ_S_WRAPPER,
                             SZ_AF_WRAPPER, DBLQUAD_Z_S_WRAPPER, XSEC_QS_S_WRAPPER, XSEC_ASY_S_WRAPPER, XSEC_QE_WRAPPER,
                             XSEC_QSI_WRAPPER, DBLQUAD_Z_AF_WRAPPER, XSEC_QS_AF_WRAPPER)
import numpy as np
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

# ---- NMAX, S and Z ----
nmax = VEC_NMAX_WRAPPER(radius, radius_type, wavelength, eps_real, eps_imag, axis_ratio, shape)

S, Z = VEC_SZ_S_WRAPPER(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, alphaDeg, betaDeg)

or_pdf = pdf = pyr.Orientation.gaussian(std=20)
SO, ZO = VEC_SZ_AF_WRAPPER(nmax, wavelength, izaDeg, vzaDeg, iaaDeg, vaaDeg, n_alpha, n_beta, or_pdf)


# ---- Integral ----
Zi = DBLQUAD_Z_S_WRAPPER(nmax, wavelength, vzaDeg, vaaDeg, alphaDeg, betaDeg, 1)
Zv = DBLQUAD_Z_S_WRAPPER(nmax, wavelength, vzaDeg, vaaDeg, alphaDeg, betaDeg, 0)

or_pdf = pdf = pyr.Orientation.gaussian(std=20)
Zia = DBLQUAD_Z_AF_WRAPPER(nmax, wavelength, vzaDeg, vaaDeg, n_alpha, n_beta, or_pdf, 1)

Qs = XSEC_QS_S_WRAPPER(nmax, wavelength, vzaDeg, vaaDeg, alphaDeg, betaDeg)
Qasy = XSEC_ASY_S_WRAPPER(nmax, wavelength, vzaDeg, vaaDeg, alphaDeg, betaDeg)
Qe = XSEC_QE_WRAPPER(S, wavelength)
Qsi = XSEC_QSI_WRAPPER(Z)

Qs = XSEC_QS_AF_WRAPPER(nmax, wavelength, vzaDeg, vaaDeg, n_alpha, n_beta, or_pdf)




from pyrism.scattering import TMatrix
import numpy as np

izaDeg = np.linspace(10, 30, 5)
vzaDeg = 30
iaaDeg = 150
vaaDeg = 100
radius = 0.03
eps = (10 + 5j)

tmat = TMatrix(iza=izaDeg, vza=vzaDeg, iaa=iaaDeg, vaa=vaaDeg, frequency=1.26, radius=radius, eps=eps)

tmat = TMatrixSingle(iza=izaDeg, vza=vzaDeg, iaa=iaaDeg, vaa=vaaDeg, frequency=1.26, radius=radius, eps=eps)
S = tmat.S
Z = tmat.Z

qs = tmat.QS
qe = tmat.QE
omega = qs.base / qe.base

qas = tmat.QAS





