# from __future__ import division
# from radarpy import Angles
#
# import numpy as np
# from pyrism.fortran_tm import fotm as tmatrix
# from pyrism.scattering.tmatrix.tauxil import get_points_and_weights, gaussian_pdf, uniform_pdf, PSDIntegrator
# from scipy.integrate import dblquad
#
#
# class TMatrix(Angles):
#     def __init__(self, iza, vza, iaa, vaa, frequency, radius, eps, radius_type='REV', axis_ratio=1.0, shape='SPH',
#                  alpha=0.0, beta=0.0, Kw_sqr=0.93, orient='S', or_pdf='gauss', n_alpha=5, n_beta=10,
#                  psd_integrator=False, psd=None, angle_unit='DEG'):
#         """T-Matrix scattering from nonspherical particles.
#
#         Class for simulating scattering from nonspherical particles with the
#         T-Matrix method. Uses a wrapper to the Fortran code by M. Mishchenko.
#
#         Usage instructions:
#
#         First, the class should be be initialized. Any attributes (see below)
#         can be passed as keyword arguments to the constructor. For example:
#         sca = tmatrix.Scatterer(wavelength=2.0, m=complex(0,2))
#
#         The properties of the scattering and the radiation should then be set
#         as attributes of this object.
#
#         The functions for computing the various scattering properties can then be
#         called. The Scatterer object will automatically recompute the T-matrix
#         and/or the amplitude and phase matrices when needed.
#
#         Parameters
#         ----------
#         iza, vza, raa, ira, vra : int, float or ndarray
#             Incidence (iza) and scattering (vza) zenith angle, incidence and viewing
#             azimuth angle (ira, vra). If raa is defined, ira and vra are not mandatory.
#         wavelength :
#             The wavelength of incident light.
#         radius :
#             Equivalent radius.
#         radius_type : {'EV', 'M', 'REA'}
#             Specifacion of radius:
#                 * 'REV': radius is the equivalent volume radius (default).
#                 * 'M': radius is the maximum radius.
#                 * 'REA': radius is the equivalent area radius.
#         eps :
#             The complex refractive index.
#         axis_ratio :
#             The horizontal-to-rotational axis ratio.
#         shape : {'SPH', 'CYL'}
#             Shape of the scatter:
#                 * 'SPH' : spheroid,
#                 * 'CYL' : cylinders.
#         alpha, beta:
#             The Euler angles of the particle orientation (degrees).
#         Kw_sqr :
#             The squared reference water dielectric factor for computing
#             radar reflectivity.
#         orient : {'S', 'AA', 'AF'}
#         The function to use to compute the scattering properties:
#             * 'S': Single (default).
#             * 'AA': Averaged Adaptive
#             * 'AF': Averaged Fixed.
#         or_pdf: {'gauss', 'uniform'}
#             Particle orientation PDF for orientational averaging:
#                 * 'gauss': Use a Gaussian PDF (default).
#                 * 'uniform': Use a uniform PDR.
#         n_alpha :
#             Number of integration points in the alpha Euler angle.
#         n_beta :
#             Umber of integration points in the beta Euler angle.
#         psd_integrator :
#             Set this to a PSDIntegrator instance to enable size
#             distribution integration. If this is None (default), size
#             distribution integration is not used. See the PSDIntegrator
#             documentation for more information.
#         psd :
#             Set to a callable object giving the PSD value for a given
#             diameter (for example a GammaPSD instance); default None. Has no
#             effect if psd_integrator is None.
#         angle_unit : {'DEG', 'RAD'}, optional
#             * 'DEG': All input angles (iza, vza, raa) are in [DEG] (default).
#             * 'RAD': All input angles (iza, vza, raa) are in [RAD].
#
#         """
#         param = {'REV': 1.0,
#                  'REA': 0.0,
#                  'M': 2.0,
#                  'SPH': -1,
#                  'CYL': -2}
#
#         super(TMatrix, self).__init__(iza=iza, vza=vza, raa=None, iaa=iaa, vaa=vaa, normalize=False,
#                                       angle_unit=angle_unit)
#
#         self.radius = radius
#         self.radius_type = param[radius_type]
#
#         self.wavelength = 29.9792458 / frequency
#         self.m = eps
#         self.axis_ratio = axis_ratio
#         self.shape = param[shape]
#         self.ddelt = 1e-3
#         self.ndgs = 2
#         self.alpha = alpha
#         self.beta = beta
#         self.Kw_sqr = Kw_sqr
#         self.orient = self.orientation(orient)
#
#         self.or_pdf = self.get_pdf(or_pdf)
#         self.n_alpha = n_alpha
#         self.n_beta = n_beta
#
#         if psd_integrator:
#             self.psd_integrator = PSDIntegrator()
#         else:
#             self.psd_integrator = None
#
#         self.psd = psd
#         self.nmax = self._init_tmatrix()
#
#     def _init_tmatrix(self):
#         """Initialize the T-matrix.
#         """
#
#         if self.radius_type == 2.0:
#             # Maximum radius is not directly supported in the original
#             # so we convert it to equal volume radius
#             radius_type = 1.0
#             radius = self.__equal_volume_from_maximum()
#         else:
#             radius_type = self.radius_type
#             radius = self.radius
#
#         return tmatrix.calctmat(radius, radius_type, self.wavelength, self.m.real, self.m.imag, self.axis_ratio,
#                                 self.shape, self.ddelt, self.ndgs)
#
#     def get_SZ_single(self):
#         """Get the S and Z matrices for a single orientation.
#         """
#         self._S_single, self._Z_single = tmatrix.calcampl(self.nmax,
#                                                           self.wavelength, self.izaDeg, self.vzaDeg, self.iaaDeg,
#                                                           self.vaaDeg,
#                                                           self.alpha, self.beta)
#
#         return self._S_single, self._Z_single
#
#     def get_SZ(self):
#         """Get the S and Z matrices using the current parameters.
#         """
#         if self.psd_integrator is None:
#             self._S, self._Z = self.get_SZ_orient()
#         else:
#             self._S, self._Z = self.psd_integrator(self.psd, self.izaDeg, self.vzaDeg, self.iaaDeg, self.vaaDeg,
#                                                    self.alpha, self.beta)
#
#         return self._S, self._Z
#
#     @property
#     def S(self):
#         return self._S
#
#     @property
#     def Z(self):
#         return self._Z
#
#     def get_SZ_orient(self):
#         """Get the S and Z matrices using the specified orientation averaging.
#         """
#         self._S_orient, self._Z_orient = self.orient()
#
#         return self._S_orient, self._Z_orient
#
#     def get_pdf(self, pdf):
#         if pdf is 'gauss':
#             return gaussian_pdf()
#         elif pdf is 'uniform':
#             return uniform_pdf()
#
#     def orientation(self, orient):
#         if orient is 'S':
#             return self.__orient_single
#         elif orient is 'AA':
#             return self.__orient_averaged_adaptive
#
#         elif orient is 'AF':
#             return self.__orient_averaged_fixed
#         else:
#             raise AttributeError("The parameter orient must be 'S', 'AA' or 'AF'")
#
#     def __orient_single(self):
#         """Compute the T-matrix using a single orientation scatterer.
#
#         Args:
#             tm: TMatrix (or descendant) instance
#
#         Returns:
#             The amplitude (S) and phase (Z) matrices.
#         """
#         return self.get_SZ_single()
#
#     def __orient_averaged_adaptive(self):
#         """Compute the T-matrix using variable orientation scatterers.
#
#         This method uses a very slow adaptive routine and should mainly be used
#         for reference purposes. Uses the set particle orientation PDF, ignoring
#         the alpha and beta attributes.
#
#         Args:
#             self: selfatrix (or descendant) instance
#
#         Returns:
#             The amplitude (S) and phase (Z) matrices.
#         """
#         S = np.zeros((2, 2), dtype=complex)
#         Z = np.zeros((4, 4))
#
#         def Sfunc(beta, alpha, i, j, real):
#             (S_ang, Z_ang) = self.get_SZ_single()
#             s = S_ang[i, j].real if real else S_ang[i, j].imag
#             return s * self.or_pdf(beta)
#
#         ind = range(2)
#         for i in ind:
#             for j in ind:
#                 S.real[i, j] = dblquad(Sfunc, 0.0, 360.0,
#                                        lambda x: 0.0, lambda x: 180.0, (i, j, True))[0] / 360.0
#                 S.imag[i, j] = dblquad(Sfunc, 0.0, 360.0,
#                                        lambda x: 0.0, lambda x: 180.0, (i, j, False))[0] / 360.0
#
#         def Zfunc(beta, alpha, i, j):
#             (S_and, Z_ang) = self.get_SZ_single()
#             return Z_ang[i, j] * self.or_pdf(beta)
#
#         ind = range(4)
#         for i in ind:
#             for j in ind:
#                 Z[i, j] = dblquad(Zfunc, 0.0, 360.0,
#                                   lambda x: 0.0, lambda x: 180.0, (i, j))[0] / 360.0
#
#         return (S, Z)
#
#     def __orient_averaged_fixed(self):
#         """Compute the T-matrix using variable orientation scatterers.
#
#         This method uses a fast Gaussian quadrature and is suitable
#         for most use. Uses the set particle orientation PDF, ignoring
#         the alpha and beta attributes.
#
#         Args:
#             self: selfatrix (or descendant) instance.
#
#         Returns:
#             The amplitude (S) and phase (Z) matrices.
#         """
#         S = np.zeros((2, 2), dtype=complex)
#         Z = np.zeros((4, 4))
#         ap = np.linspace(0, 360, self.n_alpha + 1)[:-1]
#         aw = 1.0 / self.n_alpha
#
#         self.beta_p, self.beta_w = get_points_and_weights(self.or_pdf, 0, 180, self.n_beta)
#
#         for self.alpha in ap:
#             for (beta, w) in zip(self.beta_p, self.beta_w):
#                 (S_ang, Z_ang) = self.get_SZ_single()
#                 S += w * S_ang
#                 Z += w * Z_ang
#
#         sw = self.beta_w.sum()
#         # normalize to get a proper average
#         S *= aw / sw
#         Z *= aw / sw
#
#         return (S, Z)
#
#     def __equal_volume_from_maximum(self):
#         if self.shape == -1:
#             if self.axis_ratio > 1.0:  # oblate
#                 r_eq = self.radius / self.axis_ratio ** (1.0 / 3.0)
#             else:  # prolate
#                 r_eq = self.radius / self.axis_ratio ** (2.0 / 3.0)
#         elif self.shape == -2:
#             if self.axis_ratio > 1.0:  # oblate
#                 r_eq = self.radius * (0.75 / self.axis_ratio) ** (1.0 / 3.0)
#             else:  # prolate
#                 r_eq = self.radius * (0.75 / self.axis_ratio) ** (2.0 / 3.0)
#         else:
#             raise AttributeError("Unsupported shape for maximum radius.")
#         return r_eq
