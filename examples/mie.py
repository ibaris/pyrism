import numpy as np


def M0(S, p, q, wavelength=23):
    return wavelength * S[p, q]


def M1(S, p, q, wavelength=23):
    k = 2 * np.pi / wavelength

    return 1j * 2 * np.pi / k * S[p, q]


a = np.random.random(1) + np.random.random(1) * 1j
b = np.random.random(1) + np.random.random(1) * 1j
c = np.random.random(1) + np.random.random(1) * 1j
d = np.random.random(1) + np.random.random(1) * 1j

S = np.array([[a, b], [c, d]])

# Extinction row 1
assert 2 * M0(S, 0, 0).imag == -2 * M1(S, 0, 0).real  # k11
assert M0(S, 0, 1).imag == -M1(S, 0, 1).real  # k13
assert M0(S, 0, 1).imag == -M1(S, 0, 1).real  # k14

# Extinction row 2
assert 2 * M0(S, 1, 1).imag == -2 * M1(S, 1, 1).real  # k12
assert M0(S, 1, 0).imag == -M1(S, 1, 0).real  # k13
assert M0(S, 1, 0).real == M1(S, 1, 0).imag  # k14

# Extinction row 3
assert 2 * M0(S, 1, 0).imag == -2 * M1(S, 1, 0).real  # k11
assert 2 * M0(S, 0, 1).imag == -2 * M1(S, 0, 1).real  # k12
assert np.imag((M0(S, 0, 0) + M0(S, 1, 1))) == -np.real((M1(S, 0, 0) + M1(S, 1, 1)))  # k13
assert np.real((M0(S, 0, 0) - M0(S, 1, 1))) == np.imag((M1(S, 0, 0) - M1(S, 1, 1)))  # k14

# Extinction row 4
assert 2 * M0(S, 1, 0).real == 2 * M1(S, 1, 0).imag  # k11
assert -2 * M0(S, 0, 1).real == -2 * M1(S, 0, 1).imag  # k12
assert -np.real((M0(S, 0, 0) - M0(S, 1, 1))) == -np.imag((M1(S, 0, 0) - M1(S, 1, 1)))  # k13
assert np.imag((M0(S, 0, 0) + M0(S, 1, 1))) == -np.real((M1(S, 0, 0) + M1(S, 1, 1)))  # k14
