# At first we will run the PROSPECT model. To do this we import the prism package.
import numpy as np

import pyrism as pyr

# After that we specify the sensing geometry we want to simulate
iza = 35  # Incidence zenith angle
vza = 30  # Viewing zenith angle
raa = 50  # Relative azimuth angle

ray = pyr.Rayleigh(frequency=1.26, radius=0.010, eps_p=(0.25 + 0.1j))

ray.pmatrix(iza, vza, raa, dblquad=True)
