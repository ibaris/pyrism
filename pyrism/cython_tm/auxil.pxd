# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 28.02.2019 by Ismail Baris
"""

cdef double[:] equal_volume_from_maximum(double[:] radius, double[:] axis_ratio, int shape)