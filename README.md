<h1 align="center">
  <br>
  <a href="http://elib.dlr.de/115785/"><img src="https://i.imgur.com/3nSMbDM.png" alt="PRISM" width="400"></a>
</h1>
<h4 align="center">Python bindings for Remote Sensing Models </h4>

<p align="center">
  <a href="http://forthebadge.com">
    <img src="http://forthebadge.com/images/badges/made-with-python.svg"
         alt="Gitter">
  </a>
  <a href="http://forthebadge.com"><img src="http://forthebadge.com/images/badges/built-with-love.svg"></a>
  <a href="http://forthebadge.com">
      <img src="http://forthebadge.com/images/badges/built-with-science.svg">
  </a>
</p>


<p align="center">
  <a href="#description">Description</a> •
  <a href="#installation">Installation</a> •
  <a href="#example">Example</a> •
  <a href="#authors">Author</a> •
  <a href="#acknowledgments">Acknowledgments</a> •
</p>

<p align="center">
  <a href="https://travis-ci.com/ibaris/ROM"><img src="https://travis-ci.org/ibaris/prism.svg?branch=master"></a>
  <a href='https://coveralls.io/github/ibaris/prism?branch=master'><img src='https://coveralls.io/repos/github/ibaris/prism/badge.svg?branch=master' alt='Coverage Status' /></a>

</p>

# Description
This repository contains the Python bindings to different radar and optical backscattering and reflectance models, respectively. The bindings implement the following models:
### Optical Models:
* **PROSPECT**: Leaf reflectance model (versions 5 and D).
* **SAIL**: Canopy reflectance model.
* **PROSAIL**: Combination of PROSPECT and SAIL.
* **LSM**: Simple Lambertian soil reflectance model.
* **Volume Scattering**: Compute volume scattering functions and interception coefficients for given solar zenith, viewing zenith, azimuth and leaf inclination angle.
### RADAR Models:
* **Rayleigh**: Calculate the extinction coefficients in terms of Rayleigh scattering.
* **Mie**: Calculate the extinction coefficients in terms of Mie scattering.
* **Dielectric Constants**: Calculate the dielectric constant of different objects like water, saline water, soil and vegetation.
* **I2EM**: RADAR soil scattering model to compute the backscattering coefficient VV and HH polarized.
* **Emissivity**: Calculate the emissivity for single-scale random surface for Bi and Mono-static acquisitions.

For the optical models the code from <a href="https://github.com/jgomezdans/prosail"> José Gómez-Dans</a> was used as a benchmark. The theory of the radar models is from <a href="http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7067059"> F.T. Ulaby</a>

# Installation
There are currently different methods to install `prism`.
### Using pip
The ` prism ` package is provided on pip. You can install it with::

    pip install prism
### Standard Python
You can also download the source code package from this repository or from pip. Unpack the file you obtained into some directory (it can be a temporary directory) and then run::

    python setup.py install
  
### Test installation success
Independent how you installed ` prism `, you should test that it was sucessfull by the following tests::

    python -c "from prism import I2EM"

If you don't get an error message, the module import was sucessfull.

# Example
At first we will run the PROSPECT model. To do this we import the prism package.
```python
import prism
```
After that we specify the sensing geometry we want to simulate:
```python
iza = 35  # Incidence zenith angle
vza = 30  # Viewing zenith angle
raa = 50  # Relative azimuth angle
```
Than we call the PROSPECT model:
```python
prospect = prism.PROSPECT(N=1.5,
                          Cab=35,
                          Cxc=5,
                          Cbr=0.15,
                          Cw=0.003,
                          Cm=0.0055)
```

To access the attributes there are to ways. Firstly, we can access the whole spectrum with `prospect.ks` (scattering coefficient), `prospect.ka` (absorption coefficient), `prospect.kt` (transmittance coefficient), `prospect.ke` (extinction coefficient) and `prospect.om` (the division of ks through ke). Moreover, you can select the coefficient values for the specific bands of ASTER (B1 - B9) or LANDSAT8 (B2 - B7). To access these bands type `prospect.L8.Bx` (x = 2, 3, ..., 7) for Landsat 8 or `prospect.ASTER.Bx` (x = 1, 2, ..., 9) for ASTER.

To calculate the PROSAIL model we must call SAIL and specify the scattering and transmittance coefficients with these from PROSPECT like:
```python
prosail = prism.SAIL(iza=iza, vza=vza, raa=raa, ks=prospect.ks, kt=prospect.kt, lidf_type='campbell',
                     lai=3, hotspot=0.25, soil_reflectance=3.14/4, soil_moisture=0.15)
```
The accessibility of the attributes are the same as the PROSPECT model.


# Built With
* Python 2.7 (But it works with Python 3.5 as well)
* Requirements: numpy, scipy

# Authors
* **Ismail Baris** - *Initial work* - (i.baris@outlook.de)

## Acknowledgments
* Thomas Jagdhuber

---

> ResearchGate [@Ismail_Baris](https://www.researchgate.net/profile/Ismail_Baris) &nbsp;&middot;&nbsp;
> GitHub [@ibaris](https://github.com/ibaris) &nbsp;&middot;&nbsp;
> Instagram [@ism.baris](https://www.instagram.com/ism.baris/)
