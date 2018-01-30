# -*- coding: utf-8 -*-
from .curie import *
import numpy as np
from scipy.optimize import minimize
from scipy.special import polygamma

try: range=xrange
except: pass

class CurieOptimise(CurieGrid):
    """
    Extends the CurieGrid class to include optimisation routines
    see scipy.optimize.minimize for a description of the algorithm

    Parameters
    ----------
     grid     : 2D array of magnetic data
     xmin     : minimum x bound in metres
     xmax     : maximum x bound in metres
     ymin     : minimum y bound in metres
     ymax     : maximum y bound in metres
    
    Attributes
    ----------
     bounds   : lower and upper bounds for beta, zt, dz, C
     prior    : dictionary of priors for beta, zt, dz, C
    """
    def __init__(self, grid, xmin, xmax, ymin, ymax):

        super(CurieOptimise, self).__init__(grid, xmin, xmax, ymin, ymax)

        # initialise prior dictionary
        self.reset_priors()

        # lower / upper bounds
        lb = np.zeros(4)
        ub = [None]*len(lb)
        bounds = list(zip(lb,ub))
        self.bounds = bounds

        return

    def add_prior(self, **kwargs):
        """
        Add a prior to the dictionary (tuple)

        Available priors are beta, zt, dz, C

        Usage
        -----
         add_prior(beta=(p, sigma_p))
        """

        for key in kwargs:
            if key in self.prior:
                prior = kwargs[key]
                self.prior[key] = prior
            else:
                raise ValueError("prior must be one of {}".format(self.prior.keys()))

    def reset_priors(self):
        self.prior = {'beta':None, 'zt':None, 'dz':None, 'C':None}


    def objective_routine(self, **kwargs):
        """
        Evaluate the objective routine to find the misfit with priors
        Only keys stored in self.prior will be added to the total misfit

        Example Usage
        -------------
         objective_routine(beta=2.5)

        Returns
        -------
         misfit  : sum of misfit (float)
        """
        c = 0.0

        for key in kwargs:
            val = kwargs[key]
            if key in self.prior:
                prior = self.prior[key]
                if type(prior) == tuple:
                    c += self.objective_function(val, prior[0], prior[1])
        return c

    def objective_function(self, x, x0, sigma_x0):
        """
        Objective function used in objective_routine
        """
        return np.linalg.norm((x - x0)/sigma_x0)**2


    def min_func(self, x, Phi_exp, kh):
        """
        Function to minimise

        Parameters
        ----------
         x        : array of variables [beta, zt, dz, C]
         Phi_exp  : radial spectrum
         kh       : wavenumbers [rad/km]

        Returns
        -------
         misfit   : sum of misfit (float)

        Notes
        -----
         We purposely ignore all warnings raised by the bouligand2009
         function because some combinations of input parameters will
         trigger an out-of-range warning that will crash the minimiser
         when this occurs we overwrite the misfit with a very large number
        """
        beta, zt, dz, C = x
        with warnings.catch_warnings() as w:
            warnings.simplefilter("ignore")
            Phi_syn = bouligand2009(beta, zt, dz, kh, C)

        misfit = np.linalg.norm(Phi_exp - Phi_syn)
        if not np.isfinite(misfit):
            misfit = 1e99
        else:
            misfit += self.objective_routine(beta=beta, zt=zt, dz=dz, C=C)
        return misfit


    def optimise(self, window, xc, yc, beta=3.0, zt=1.0, dz=20.0, C=5.0, taper=np.hanning, process_subgrid=None, **kwargs):
        """
        Find the optimal parameters of beta, zt, dz, C for a given
        centroid (xc,yc) and window size.

        Parameters
        ----------
         window  : size of window in metres
         xc_list : centroid x values shape(l,)
         yc_list : centroid y values shape(l,)
         beta    : fractal parameter (starting value)
         zt      : top of magnetic layer (starting value)
         dz      : thickness of magnetic layer (starting value)
         C       : field constant (starting value)

        Returns
        -------
         beta    : fractal parameters
         zt      : top of magnetic layer
         dz      : thickness of magnetic layer
         C       : field constant
        """

        if type(process_subgrid) == type(None):
            # dummy function
            def process_subgrid(subgrid):
                return subgrid

        # initial constants for minimisation
        # w = 1.0 # weight low frequency?

        x0 = np.array([beta, zt, dz, C])

        # get subgrid
        subgrid = self.subgrid(xc, yc, window)
        subgrid = process_subgrid(subgrid)

        # compute radial spectrum
        S, k, sigma2 = self.radial_spectrum(subgrid, taper=taper, **kwargs)

        # minimise function
        res = minimize(self.min_func, x0, args=(S, k), bounds=self.bounds)
        return res.x


    def optimise_routine(self, window, xc_list, yc_list, beta=3.0, zt=1.0, dz=20.0, C=5.0, taper=np.hanning, process_subgrid=None, **kwargs):
        """
        Iterate through a list of centroids to compute the optimal values
        of beta, zt, dz, C for a given window size.

        Use this routine to iteratively improve the precision of various
        parameters (see notes)

        CAUTION! Priors will be altered at the end of this routine!
        
        Parameters
        ----------
         window  : size of window in metres
         xc_list : centroid x values shape(l,)
         yc_list : centroid y values shape(l,)
         beta    : fractal parameter shape(l,)
         zt      : top of magnetic layer shape(l,)
         dz      : thickness of magnetic layer shape(l,)
         C       : field constant shape(l,)

        Returns
        -------
         beta    : fractal parameters shape(l,)
         zt      : top of magnetic layer shape(l,)
         dz      : thickness of magnetic layer shape(l,)
         C       : field constant shape(l,)

        Notes
        -----
         Parameters such as beta and C vary over long wavelengths,
         thus keeping these somewhat rigid can improve the precision
         of zt and dz.
         The mean and stdev of any vectors for beta, zt, dz, C can
         be added as priors in the objective routine using add_prior

         Recommended usage is two passes:
         1. keep beta, zt, dz, C set to None with no prior
         2. add the mean and stdev of beta, zt, dz, C as priors
            and run through a second pass
        """

        n = len(xc_list)
        
        if n != len(yc_list):
            raise ValueError("xc_list and yc_list must be the same size")


        # storage vectors
        Bopt  = np.empty(n)
        ztopt = np.empty(n)
        dzopt = np.empty(n)
        Copt  = np.empty(n)

        for i in range(0, n):
            xc = xc_list[i]
            yc = yc_list[i]

            Bopt[i], ztopt[i], dzopt[i], Copt[i] = self.optimise(window, xc, yc,\
                                                                 beta, zt, dz, C,\
                                                                 taper, process_subgrid,\
                                                                 **kwargs)

        return Bopt, ztopt, dzopt, Copt


    def _prioritise(self, mask, **kwargs):
        """
        Calculate mean and stdev for beta, zt, dz, C that were provided
        in optimise_routine and set as priors.

        mask selects entries that are within the window size

        Returns
        ----------
         x0   : initial x variables
        """
        self.reset_priors()
        x0 = np.array([3.0, 1.0, 20.0, 5.0])

        for i, key in enumerate(['beta', 'zt', 'dz', 'C']):
            array = kwargs[key]
            if type(array) == np.ndarray or type(array) == list:
                marray = array[mask]
                v  = np.mean(marray)
                dv = np.std(marray)
                if dv != 0.0:
                    self.prior[key] = (v, dv)
                    x0[i] = v

        return x0
