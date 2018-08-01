# At first we will run the PROSPECT model. To do this we import the prism package.
import numpy as np

import pyrism as pyr

# After that we specify the sensing geometry we want to simulate
iza = 35  # Incidence zenith angle
vza = 30  # Viewing zenith angle
raa = 50  # Relative azimuth angle

ray = pyr.Rayleigh(frequency=1.26, particle_size=0.010, diel_constant_p=(0.25 + 0.1j))

ray.phase_matrix(iza, vza, raa, normalize=False, integrate=True)

mat = pyr.Rayleigh.phase_matrix(iza, vza, raa)

iza = np.arange(30, 35, 1)  # Incidence zenith angle
vza = np.arange(25, 30, 1)  # Viewing zenith angle
raa = np.arange(40, 45, 1)  # Relative azimuth angle

ray.phase_matrix(iza, vza, raa, normalize=False, integrate=True)
