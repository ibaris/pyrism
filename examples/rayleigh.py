# At first we will run the PROSPECT model. To do this we import the prism package.
from __future__ import division
import numpy as np

import pyrism as pyr

# After that we specify the sensing geometry we want to simulate
iza = np.arange(10, 30, 1)  # Incidence zenith angle
vza = 30  # Viewing zenith angle
raa = 50  # Relative azimuth angle

ray = pyr.Rayleigh(frequency=1.26, radius=np.arange(0, 0.0205, 0.0005), eps_p=(0.25 + 0.1j))

pmatrix = pyr.Rayleigh.Phase(20, vza, raa)
print pmatrix

pmatrix.quad()
pmatrix.quad(precalc=False)
pmatrix.dblquad()
pmatrix.dblquad(precalc=False)
