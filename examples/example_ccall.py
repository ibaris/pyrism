import pyrism as pyr
import numpy as np

iza = np.arange(0, 80, 1)
freq = np.arange(1, 3.1, 0.1)
a = pyr.I2EM(10, 30, 50, frequency=freq, eps=(6.9076590988636735 + 0.55947861142615318j), corrlength=20, sigma=.3)
b = pyr.I2EM.Emissivity(10, frequency=freq, eps=(6.9076590988636735 + 0.55947861142615318j), corrlength=20, sigma=.3)
