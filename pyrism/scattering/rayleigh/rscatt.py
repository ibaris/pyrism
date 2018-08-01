import sys
import warnings

import numpy as np

from phase.phase_c import phase_matrix
from ...core import Scattering, Kernel

# python 3.6 comparability
if sys.version_info < (3, 0):
    srange = xrange
else:
    srange = range


class Rayleigh(Scattering):
    """
    Calculate the extinction coefficients in terms of Rayleigh
    scattering (:cite:`Ulaby.2015` and :cite:`Ulaby.2015b`).

    Parameters
    ----------
    frequency : int or float
        Frequency (GHz)
    particle_size : int, float or array
        Particle size a [m].
    diel_constant_p : complex
        Dielectric constant of the medium.
    diel_constant_b : complex
        Dielectric constant of the background.

    Returns
    -------
    All returns are attributes!
    self.ke : int, float or array_like
        Extinction coefficient.
    self.ks : int, float or array_like
        Scattering coefficient.
    self.ka : int, float or array_like
        Absorption coefficient.
    self.om : int, float or array_like
        Omega.
    self.s0 : int, float or array_like
        Backscatter coefficient sigma 0.

    """

    def __init__(self, frequency, particle_size, diel_constant_p, diel_constant_b=(1 + 1j)):

        super(Rayleigh, self).__init__(frequency, particle_size, diel_constant_p, diel_constant_b)

        # Check validity
        lm = 299792458 / (self.freq * 1e9)  # Wavelength in meter
        self.condition = (2 * np.pi * self.a) / lm

        if np.any(self.condition >= 0.5):
            warnings.warn("Rayleigh condition not holds. You should use Mie scattering.", Warning)
        else:
            pass

        self.__calc()

    def __calc(self):
        bigK = (self.n ** 2 - 1) / (self.n ** 2 + 2)
        self.ks = (8 / 3) * self.chi ** 4 * np.abs(bigK) ** 2
        self.ka = 4 * self.chi * (-bigK.imag)
        self.ke = self.ka + self.ks
        self.kt = 1 - self.ke
        self.BSC = 4 * self.chi ** 4 * np.abs(bigK) ** 2
        self.omega = self.ks / self.ke

    @staticmethod
    def phase_matrix(iza, vza, raa, integrate=False, normalize=False, nbar=0.0, angle_unit='DEG'):
        kernel = Kernel(iza, vza, raa, normalize=normalize, nbar=nbar, angle_unit=angle_unit)

        kernel.iza = kernel.iza
        kernel.vza = kernel.vza
        kernel.raa = kernel.raa

        if integrate is True:
            if len(kernel.iza) == 1:
                return phase_matrix(kernel.iza[0], kernel.vza[0], kernel.raa[0], 1)

            else:
                result = list()
                for x in srange(len(kernel.iza)):
                    result.append(phase_matrix(kernel.iza[x], kernel.vza[x], kernel.raa[x], 1))

                return result

        elif integrate is False:
            if len(kernel.iza) == 1:
                return phase_matrix(kernel.iza[0], kernel.vza[0], kernel.raa[0], 0)

            else:
                result = list()
                for x in srange(len(kernel.iza)):
                    result.append(phase_matrix(kernel.iza[x], kernel.vza[x], kernel.raa[x], 0))

                return result
        else:
            raise AssertionError("The parameter integrate must be true or false.")
