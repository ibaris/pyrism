# To run the soil module we have to import pyrism.
# Optional we could import it as `from pyrism import soil`

import pyrism as pyr
import numpy as np

# We import matplotlib to plot the results.
import matplotlib.pyplot as plt

# After that we specify the sensing geometry we want to simulate:
iza = np.arange(30, 35, 1)  # Incidence zenith angle
vza = 30  # Viewing zenith angle
raa = 50  # Relative azimuth angle

n1 = 1 + 0j  # Refractive index of medium 1 (Air)
n2 = 0.80 + 0j  # Refractive index of boundary medium (Soil)

frequency = 1.26  # Frequency in GHz
eps = 5 + 0.5j  # Dielectric constant of soil
corrlength = 10  # Correlation length in cm
sigma = 0.5  # RMS Height in cm

# At first we will simulate the Fresnel reflectivity.
# fresnel = pyr.Fresnel(iza=iza, frequency=frequency, n1=n1, n2=n2, sigma=sigma)

# Now we we will calculate the BSC with the I2EM Model.
# i2em = pyr.I2EM(iza, vza, raa, frequency=frequency, eps=eps, corrlength=corrlength, sigma=sigma)

# Now an optical model
lsm = pyr.LSM(0.01, 0.25)
