# -*- coding: utf-8 -*-
from .curie import *
import numpy as np
from scipy.optimize import minimize

try: range=xrange
except: pass

class CurieOptimise(CurieGrid):

    def __init__(self, grid, xmin, xmax, ymin, ymax):

        super(CurieOptimise, self).__init__(grid, xmin, xmax, ymin, ymax)

        # initialise prior dictionary
        self.reset_priors()

        # lower / upper bounds
        lb = np.zeros(4)
        ub = [None]*len(lb)
        bounds = zip(lb,ub)
        self.bounds = bounds

        return

    def add_prior(self, **kwargs):

        for key in kwargs:
            if key in self.prior:
                prior = kwargs[key]
                self.prior[key] = prior
            else:
                raise ValueError("prior must be one of {}".format(self.prior.keys()))

    def reset_priors(self):
        self.prior = {'beta':None, 'zt':None, 'dz':None, 'C':None}


    def objective_routine(self, **kwargs):

        c = 0.0

        for key in kwargs:
            val = kwargs[key]
            if key in self.prior:
                prior = self.prior[key]
                if type(prior) == tuple:
                    c += self.objective_function(val, prior[0], prior[1])
        return c

    def objective_function(self, x, x0, sigma_x0):
        return np.sum((x - x0)**2/sigma_x0**2)


    def min_func(self, x, Phi_exp, kh):
        """
        Function to minimise
        """
        beta = x[0]
        zt = x[1]
        dz = x[2]
        C = x[3]
        misfit = np.linalg.norm(w*(Phi_exp - bouligand2009(beta, zt, dz, kh, C)))
        misfit += self.objective_routine(beta=beta, zt=zt, dz=dz, C=C)
        return misfit


    def optimise(self, window, xc, yc, beta=3.0, zt=1.0, dz=20.0, C=5.0, taper=np.hanning, process_subgrid=None, **kwargs):
        """
        Find the optimal parameters of beta, zt, dz, C for a given
        centroid (xc,yc) and window size.

        """

        if type(process_subgrid) == type(None):
            # dummy function
            def process_subgrid(subgrid):
                return subgrid

        # initial constants for minimisation
        w = 1.0 # weight low frequency?

        x0 = np.array([beta, zt, dz, C])

        # get subgrid
        subgrid = self.subgrid(xc, yc, window)
        subgrid = process_subgrid(subgrid)

        # compute radial spectrum
        S, k, sigma2 = self.radial_spectrum(subgrid, taper=taper, **kwargs)

        # minimise function
        res = minimize(self.min_func, x0, args=(S, k), bounds=self.bounds)
        return res.x


    def optimise_routine(self, window, xc_list, yc_list, beta=None, zt=None, dz=None, C=None, taper=np.hanning, process_subgrid=None, **kwargs):
        """
        Iterate through a list of centroids to compute the optimal values
        of beta, zt, dz, C for a given window size.

        Use this routine to iteratively improve the precision of various
        parameters (see notes)

        CAUTION! Priors will be different at the end of this routine!
        
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
	 beta    : fractal parameters shape(l,)
	 zt      : top of magnetic layer shape(l,)
	 dz      : thickness of magnetic layer shape(l,)
	 C       : field constant shape(l,)

        Notes
        -----
         Parameters such as beta and C vary over long wavelengths,
         thus keeping these somewhat rigid can improve the precision
         of zt and dz.
         The mean and stdev of any vectors for beta, zt, dz, C will
         be added as priors in the objective routine.

         Recommended usage is two passes:
         1. keep beta, zt, dz, C set to None with no prior
         2. provide these vectors in the second pass
        """
        
        if type(process_subgrid) == type(None):
            # dummy function
            def process_subgrid(subgrid):
                return subgrid

        n = len(xc_list)
        
        if n != len(yc_list):
            raise ValueError("xc_list and yc_list must be the same size")


        # storage vectors
        Bopt  = np.zeros(n)
        ztopt = np.zeros(n)
        dzopt = np.zeros(n)
        Copt  = np.zeros(n)
        
        d2w = window/2

        for i in range(0, n):
            xc = xc_list[i]
            yc = yc_list[i]

            # find all values of beta, zt, dz, C inside the window
            xcmask = np.logical_and(xc_list >= xc-d2w, xc_list <= xc+d2w)
            ycmask = np.logical_and(yc_list >= yc-d2w, yc_list <= yc+d2w)
            lcmask = np.logical_and(xcmask, ycmask)

            # update priors
            x0 = self._prioritise(lcmask, beta=beta, zt=zt, dz=dz, C=C)
            B0, zt0, dz0, C0 = x0

            Bopt[i], ztopt[i], dzopt[i], Copt[i] = self.optimise(window, xc, yc,\
                                                                 B0, zt0, dz0, C0,\
                                                                 taper, process_subgrid,\
                                                                 **kwargs)

        return Bopt, ztopt, dzopt, Copt


    def _prioritise(self, mask, **kwargs):

        x0 = np.array([3.0, 1.0, 20.0, 5.0])

        for i, key in enumerate(['beta', 'zt', 'dz', 'C']):
            array = kwargs[key]
            if type(array) == np.ndarray or type(array) == list:
                marray = array[mask]
                v  = np.mean(marray)
                dv = np.std(marray)
                self.prior[key] = (v, dv)
                x0[i] = v

        return x0

