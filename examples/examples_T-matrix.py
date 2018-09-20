import pyrism as pyr
import numpy as np
import matplotlib.pyplot as plt

beta = np.arange(0, 181, 1)
r = np.arange(0, 4, .1)

orient = pyr.Orientation.gaussian(90, 2)
uniform = pyr.Orientation.uniform()

plt.plot(beta, orient(beta))
plt.plot(beta, uniform(beta))
plt.show()

psd = pyr.PSD(r0=1, mu=4, n0=1e3, rmax=5)
exponential = psd.exponential(r)
gamma = psd.gamma(r)

plt.plot(r, exponential)
plt.plot(r, gamma)
plt.show()
