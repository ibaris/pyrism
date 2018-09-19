import numpy as np
from pytmatrix.tmatrix import TMatrix, Scatterer
from pytmatrix.tmatrix_psd import TMatrixPSD
from pytmatrix import orientation
from pytmatrix import radar
from pytmatrix import refractive
from pytmatrix import tmatrix_aux
from pytmatrix import psd
from pytmatrix import scatter

tm = Scatterer(radius=4.0, wavelength=6.5, m=complex(1.5, 0.0),
               axis_ratio=1.0 / 0.6)
tm.set_geometry(tmatrix_aux.geom_horiz_forw)
ssa_h = scatter.ssa(tm, True)
ssa_v = scatter.ssa(tm, False)
