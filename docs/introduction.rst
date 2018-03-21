Introduction
============
This repository contains the Python bindings to different radar and optical backscattering and reflectance models, respectively. The bindings implement the following models:

Optical Models:
---------------
* **PROSPECT**: Leaf reflectance model (versions 5 and D).
* **SAIL**: Canopy reflectance model.
* **PROSAIL**: Combination of PROSPECT and SAIL.
* **LSM**: Simple Lambertian soil reflectance model.
* **Volume Scattering**: Compute volume scattering functions and interception coefficients for given solar zenith, viewing zenith, azimuth and leaf inclination angle.

RADAR Models:
-------------
* **Rayleigh**: Calculate the extinction coefficients in terms of Rayleigh scattering.
* **Mie**: Calculate the extinction coefficients in terms of Mie scattering.
* **Dielectric Constants**: Calculate the dielectric constant of different objects like water, saline water, soil and vegetation.
* **I2EM**: RADAR soil scattering model to compute the backscattering coefficient VV and HH polarized.
* **Emissivity**: Calculate the emissivity for single-scale random surface for Bi and Mono-static acquisitions.

For the optical models the code from <a href="https://github.com/jgomezdans/prosail"> José Gómez-Dans</a> was used as a benchmark. The theory of the radar models is from <a href="http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7067059"> F.T. Ulaby</a>
