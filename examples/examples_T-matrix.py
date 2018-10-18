import pyrism as pyr
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

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

# iza = np.arange(20, 50, 5)
iza = 35
vza = 20
iaa = 150
vaa = 100

tm = pyr.TMatrix(iza=iza, vza=vza, iaa=100, vaa=vaa,
                 radius=5, frequency=1.26, eps=complex(15, 0.5), axis_ratio=1 / 0.6)

tm.get_ksx()

tm.ksx
dblquad(tm.TM.ifunc_SZ, 0, 2 * np.pi, lambda x: 0.0, lambda x: np.pi, args=(1,))

print tm.TM.dblquad()

PSD = pyr.PSD()
psd = PSD.exponential

tm = pyr.TMatrix(radius=1, frequency=1.26, eps=(5 + 0.25j), psd=psd, iza=iza, vza=vza, iaa=iaa, vaa=vaa,
                 axis_ratio=1 / 0.6,
                 max_radius=10,
                 num_points=10,
                 angular_integration=True)
