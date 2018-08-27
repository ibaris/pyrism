import pyrism as pyr
import numpy as np
from pyrism.soil.fresnel.fresnel import snell_wrapper

n1 = 5 + 0.3j
n2 = 3 + 0.2j
iza = np.arange(0, 35, 1)

surface = pyr.Fresnel(iza, 1.26, n1, n2, 0.5, isometric=False)
surface.quad()
