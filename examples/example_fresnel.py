# To run the soil module we have to import pyrism.
# Optional we could import it as `from pyrism import soil`

import pyrism as pyr
import numpy as np

# We import matplotlib to plot the results.
import matplotlib.pyplot as plt

# After that we specify the sensing geometry we want to simulate:
iza = np.arange(0, 90, 1)  # Incidence zenith angle
vza = 30  # Viewing zenith angle
raa = 50  # Relative azimuth angle

frequency = 1.26  # Frequency in GHz
eps = [1, 2 + 4j, 3 + 0.3j, 1 + 0.1j]  # Dielectric constant of soil
corrlength = 10  # Correlation length in cm
sigma = 0.5  # RMS Height in cm

# At first we will simulate the Fresnel reflectivity.
fresnel = pyr.Fresnel(xza=iza, frequency=frequency, eps=eps, sigma=sigma)

# And now the emissivity.
fresnel = pyr.Fresnel.Emissivity(xza=iza, frequency=frequency, eps=eps, sigma=sigma)

# Now we we will calculate the BSC with the I2EM Model.
i2em = pyr.I2EM(iza, vza, raa, frequency=frequency, eps=eps, corrlength=corrlength, sigma=sigma)

# Now an optical model
lsm = pyr.LSM(0.01, 0.25)

plt.plot(fresnel.I.HH)
plt.show()
