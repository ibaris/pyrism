from __future__ import division

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
from scipy.special import gamma


class PSD(object):
    def __init__(self, n0=1.0, ilambda=1.0, rmax=None, r0=0.5, mu=0.0, normalize=True, edges=None, psd=None):
        """
        Callable class to provide different particle size distribution (PSD) functions.

        Parameters
        ----------
        n0 : float
            The intercept parameter. Default is 1.0
        ilambda : float
            The inverse scale parameter. Default is 1.0
        rmax : int, float or None.
            Maximum diameter to consider. If None (default) rmax will be approximated by the PSD functions.
        r0 : float
            The median volume radius. Default is 0.5. This is only recognized by PSD.gamma PSD.
        mu : float
            The shape parameter. Default is 0.0. This is only recognized by PSD.gamma PSD.
        normalize : bool
            If True (default) the normalized gamma function will be calculated. This is only recognized by PSD.gamma PSD.
        edges : array_like
            n bin edges. This is only recognized by PSD.binned PSD.
        psd : array_like
            n+1 psd values. This is only recognized by PSD.binned PSD.

        """
        self.n0 = n0
        self.ilambda = ilambda
        self.rmax = rmax
        self.r0 = r0
        self.mu = mu
        self.normalize = normalize
        self.edges = edges
        self.psd = psd

    def exponential(self, r):
        """Exponential particle size distribution (PSD).

        Method to provide an exponential PSD with the given
        parameters. The attributes can also be given as arguments to the
        constructor.

        Parameters
        ----------
        r : int or float
            Radius of particle.

        Returns
        -------
        PSD : array_like
            The PSD value for the given diameter. Returns 0 for all diameters larger than D_max.

        Note
        ----
        If rmax is None the maximum diameter will be approximated by: 11/ilambda
        """

        D = r * 2

        D_max = 11.0 / self.ilambda if self.rmax is None else self.rmax * 2

        psd = self.n0 * np.exp(-self.ilambda * D)

        if np.shape(D) == ():
            if D > D_max:
                return 0.0
        else:
            psd[D > D_max] = 0.0

        return psd

    def gamma(self, r):
        """Normalized gamma particle size distribution (PSD).

        Method to provide a normalized gamma PSD with the given
        parameters. The attributes can also be given as arguments to the
        constructor.

        The PSD form of normalized form is:
        N(D) = Nw * f(mu) * (D/D0)**mu * exp(-(3.67+mu)*D/D0)
        f(mu) = 6/(3.67**4) * (3.67+mu)**(mu+4)/Gamma(mu+4)

        The PSD form of NOT normalized form is:
        N(D) = N0 * D**mu * exp(-Lambda*D)

        Parameters
        ----------
        r : int or float
            Radius of particle.

        Returns
        -------
        PSD : array_like
            The PSD value for the given diameter. Returns 0 for all diameters larger than D_max.

        Note
        ----
        If rmax is None the maximum diameter will be approximated by: 3 * r0 * 2 for normlalized and 11/ilambda for
        NOT normalized case.
        """
        D = r * 2

        if self.normalize:
            nf = self.n0 * 6.0 / 3.67 ** 4 * (3.67 + self.mu) ** (self.mu + 4) / gamma(self.mu + 4)

            D0 = self.r0 * 2
            D_max = 3.0 * D0 if self.rmax is None else self.rmax * 2

            d = (D / D0)
            psd = nf * np.exp(self.mu * np.log(d) - (3.67 + self.mu) * d)
            if np.shape(D) == ():
                if (D > D_max) or (D == 0.0):
                    return 0.0
            else:
                psd[(D > D_max) | (D == 0.0)] = 0.0

        else:
            D_max = 11.0 / self.ilambda if self.rmax is None else self.rmax * 2
            psd = self.n0 * np.exp(self.mu * np.log(D) - self.ilambda * D)
            if np.shape(D) == ():
                if (D > D_max) or (D == 0):
                    return 0.0
            else:
                psd[(D > D_max) | (D == 0)] = 0.0
            return psd

        return psd

    def binned(self, r):
        """
        Binned PSD function.

        Parameters
        ----------
        r : int or float
            Radius of particle.

        Returns
        -------
        PSD : array_like
            The PSD value for the given diameter.
            Returns 0 for all diameters outside the bins.
        """
        D = r * 2

        if len(self.edges) != len(self.psd) + 1:
            raise ValueError("There must be n+1 bin edges for n bins.")

        if not (self.edges[0] < D <= self.edges[-1]):
            return 0.0

        # binary search for the right bin
        start = 0
        end = len(self.edges)
        while end - start > 1:
            half = (start + end) // 2
            if self.edges[start] < D <= self.edges[half]:
                end = half
            else:
                start = half

        return self.psd[start]
