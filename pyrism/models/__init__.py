from .library import get_data_one, get_data_two
from .models import (VolScatt, LIDF, PROSPECT, Rayleigh, Mie, DielConstant, CorrFunc, exponential, gaussian, xpower,
                     I2EM, LSM, SAIL)

try:
    lib = get_data_two()
except IOError:
    lib = get_data_one()
