<h1 align="center">
  <br>
  <a href="http://elib.dlr.de/115785/"><img src="https://i.imgur.com/neC2rZh.png" alt="PYRISM" width="400"></a>
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
    <a href="#documentation">Doumentation</a> •
  <a href="#authors">Author</a> •
  <a href="#acknowledgments">Acknowledgments</a>
</p>

<p align="center">
  <a href="https://www.travis-ci.org/ibaris/pyrism"><img src="https://www.travis-ci.org/ibaris/pyrism.svg?branch=master"></a>
  <a href='https://coveralls.io/github/ibaris/pyrism?branch=master'><img src='https://coveralls.io/repos/github/ibaris/pyrism/badge.svg?branch=master' alt='Coverage Status' /></a>
  <a href='http://pyrism.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/pyrism/badge/?version=latest' alt='Documentation Status' /></a>
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

For the optical models the code from <a href="https://github.com/jgomezdans/prosail"> José Gómez-Dans</a> was used as a benchmark. The theory of the radar models is from <a href="http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7067059"> F.T. Ulaby</a>.

# Installation
There are currently different methods to install `pyrism`.
### Using pip
The ` pyrism ` package is provided on pip. You can install it with::

    pip install pyrism
### Standard Python
You can also download the source code package from this repository or from pip. Unpack the file you obtained into some directory (it can be a temporary directory) and then run::

    python setup.py install
  
### Test installation success
Independent how you installed ` pyrism `, you should test that it was sucessfull by the following tests::

    python -c "from pyrism import I2EM"

If you don't get an error message, the module import was sucessfull.

# Example
At first we will run the PROSPECT model. To do this we import the pyrism package.
```python
import pyrism
```
After that we specify the sensing geometry we want to simulate:
```python
iza = 35  # Incidence zenith angle
vza = 30  # Viewing zenith angle
raa = 50  # Relative azimuth angle
```
Than we call the PROSPECT model:
```python
prospect = pyrism.PROSPECT(N=1.5,
                          Cab=35,
                          Cxc=5,
                          Cbr=0.15,
                          Cw=0.003,
                          Cm=0.0055)
```

To access the attributes there are to ways. Firstly, we can access the whole spectrum with `prospect.ks` (scattering coefficient), `prospect.ka` (absorption coefficient), `prospect.kt` (transmittance coefficient), `prospect.ke` (extinction coefficient) and `prospect.om` (the division of ks through ke). Moreover, you can select the coefficient values for the specific bands of ASTER (B1 - B9) or LANDSAT8 (B2 - B7). To access these bands type `prospect.L8.Bx` (x = 2, 3, ..., 7) for Landsat 8 or `prospect.ASTER.Bx` (x = 1, 2, ..., 9) for ASTER.

To calculate the PROSAIL model we need some soil reflectance values. To obtain these we can use the LSM model:
```python
lsm = pyrism.LSM(reflectance=3.14 / 4, moisture=0.15)
```

Now we must call SAIL and specify the scattering and transmittance coefficients with these from PROSPECT like:
```python
prosail = pyrism.SAIL(iza=iza, vza=vza, raa=raa, ks=prospect.ks, kt=prospect.kt, rho_surface=lsm.ref,
                      lidf_type='campbell',
                      lai=3, hotspot=0.25)
```

The accessibility of the attributes are the same as the PROSPECT model.

# Documentation
You can find the full documentation <a href="http://pyrism.readthedocs.io/en/latest/index.html">here</a>.

# Built With
* Python 2.7 (But it works with Python 3.5 as well)
* Requirements: numpy, scipy

# Authors
* **Ismail Baris** - *Initial work* - (i.baris@outlook.de)

## Acknowledgments
*  <a href="https://www.researchgate.net/profile/Thomas_Jagdhuber">Thomas Jagdhuber </a>

---

> ResearchGate [@Ismail_Baris](https://www.researchgate.net/profile/Ismail_Baris) &nbsp;&middot;&nbsp;
> GitHub [@ibaris](https://github.com/ibaris) &nbsp;&middot;&nbsp;
> Instagram [@ism.baris](https://www.instagram.com/ism.baris/)
