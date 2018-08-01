import numpy as np


class Scattering(object):
    """

    Calculate the extinction coefficients in terms of Rayleigh or Mie
    scattering from :cite:`Ulaby.2015` and :cite:`Ulaby.2015b`.

    Parameters
    ----------
    frequency : int or float
        Frequency (GHz)

    particle_size : int, float or array
        Particle size a (cm).

    diel_constant_p : complex
        Dielectric constant of the medium.

    diel_constant_b : complex
        Dielectric constant of the background.

    """

    def __init__(self, frequency, particle_size, diel_constant_p, diel_constant_b):
        frequency, particle_size, diel_constant_p, diel_constant_b = asarrays(
            (frequency, particle_size, diel_constant_p, diel_constant_b))

        self.freq = frequency
        self.a = particle_size
        self.er_p = diel_constant_p
        self.er_b = diel_constant_b
        self.__pre_process()

    def __pre_process(self):
        self.er_b_real = self.er_b.real
        self.np = np.sqrt(self.er_p)  # index of refraction of spherical particle
        self.nb = np.sqrt(self.er_b)  # index of refraction of background medium
        self.n = self.np / self.nb
        self.chi = (20 / 3) * np.pi * self.a * self.freq * np.sqrt(self.er_b_real)
