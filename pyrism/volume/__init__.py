from library import get_data_one, get_data_two
from volume import (VolScatt, LIDF, PROSPECT, SAIL)

try:
    lib = get_data_two()
except IOError:
    lib = get_data_one()
