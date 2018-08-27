import numpy as np
from radarpy import cot
import cmath
from .auxiliary import end_sum


def mie_scattering(frequency, radius, eps_p, eps_b):
    eps_b_real = eps_b.real

    n_p = cmath.sqrt(eps_p)
    n_b = cmath.sqrt(eps_b)
    n = n_p / n_b

    chi = 20.0 / 3.0 * np.pi * radius * frequency * np.sqrt(eps_b_real)
    BigK = ((n * n) - 1) / ((n * n) + 2)

    l = 1
    first = True
    runSum = np.zeros_like(frequency)
    oldSum = np.zeros_like(frequency)

    W1 = np.sin(chi) + 1j * np.cos(chi)
    W2 = np.cos(chi) - 1j * np.sin(chi)
    A1 = cot(n * chi)

    try:
        lenchi = len(chi)
    except:
        lenchi = 1

    while first or not end_sum(oldSum, runSum, lenchi):
        W = (2 * l - 1) / chi * W1 - W2

        A = -l / (n * chi) + (l / (n * chi) - A1) ** (-1)

        a = ((A / n + l / chi) * W.real - W1.real) / ((A / n + l / chi) * W - W1)
        b = ((n * A + l / chi) * W.real - W1.real) / ((n * A + l / chi) * W - W1)

        sumTerm = (2 * l + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2)
        oldSum = runSum
        runSum = runSum + sumTerm

        l += 1

        W2 = W1
        W1 = W

        A1 = A

        first = False

    oldSum = oldSum
    runSum = runSum

    ks = 2 / chi ** 2 * runSum

    l = 1
    first = True
    runSum = np.zeros_like(frequency)
    oldSum = np.zeros_like(frequency)

    W1 = np.sin(chi) + 1j * np.cos(chi)
    W2 = np.cos(chi) - 1j * np.sin(chi)
    A1 = cot(n * chi)

    while first or not end_sum(oldSum, runSum, lenchi):
        W = (2 * l - 1) / chi * W1 - W2
        A = -l / (n * chi) + (l / (n * chi) - A1) ** (-1)

        a = ((A / n + l / chi) * W.real - W1.real) / ((A / n + l / chi) * W - W1)
        b = ((n * A + l / chi) * W.real - W1.real) / ((n * A + l / chi) * W - W1)

        sumTerm = (2 * l + 1) * np.real(a + b)
        oldSum = runSum
        runSum = runSum + sumTerm

        l += 1

        W2 = W1
        W1 = W

        A1 = A

        first = False

    ke = 2 / chi ** 2 * runSum
    omega = ks / ke
    ka = ke - ks
    kt = 1 - ke

    l = 1
    first = True
    runSum = np.zeros_like(frequency)
    oldSum = np.zeros_like(frequency)

    W1 = np.sin(chi) + 1j * np.cos(chi)
    W2 = np.cos(chi) - 1j * np.sin(chi)
    A1 = cot(n * chi)

    while first or not end_sum(oldSum, runSum, lenchi):
        W = (2 * l - 1) / chi * W1 - W2
        A = -l / (n * chi) + (l / (n * chi) - A1) ** (-1)

        a = ((A / n + l / chi) * W.real - W1.real) / ((A / n + l / chi) * W - W1)
        b = ((n * A + l / chi) * W.real - W1.real) / ((n * A + l / chi) * W - W1)

        sumTerm = (-1) ** l * (2 * l + 1) * (a - b)
        oldSum = runSum
        runSum = runSum + sumTerm

        l += 1

        W2 = W1
        W1 = W

        A1 = A

        first = False

    bsc = 1 / chi ** 2 * np.abs(runSum) ** 2

    return ks, ka, kt, ke, omega, bsc
