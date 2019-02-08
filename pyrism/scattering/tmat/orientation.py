from __future__ import division

from pyrism.core.tma import uniform_wrapper, gaussian_wrapper


class Orientation(object):
    def __init__(self):
        pass

    @staticmethod
    def gaussian(std=10.0, mean=0.0):
        """Gaussian probability distribution function (PDF) for orientation averaging.

        Parameters
        ----------
        std: int or float
            The standard deviation in degrees of the Gaussian PDF
        mean: int or float
            The mean in degrees of the Gaussian PDF.  This should be a number in the interval [0, 180)

        Returns
        -------
        pdf(x): callable
            A function that returns the value of the spherical Jacobian-normalized Gaussian PDF with the given STD at x
            (degrees). It is normalized for the interval [0, 180].
        """
        return gaussian_wrapper(std, mean)

    @staticmethod
    def uniform():
        """Uniform probability distribution function (PDF) for orientation averaging.

        Returns
        -------
        pdf(x): callable
            A function that returns the value of the spherical Jacobian-normalized uniform PDF. It is normalized for
            the interval [0, 180].
        """
        return uniform_wrapper()
