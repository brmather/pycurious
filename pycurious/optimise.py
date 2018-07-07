# -*- coding: utf-8 -*-
from .curie import *
import numpy as np
from scipy.optimize import minimize
from scipy.special import polygamma
from scipy import stats

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
        # [beta, zt, dz, C]
        lb = [0.0, 0.0, 0.0, None]
        ub = [None]*len(lb)
        bounds = list(zip(lb,ub))
        self.bounds = bounds

        return

    def add_prior(self, **kwargs):
        """
        Add a prior to the dictionary (tuple)
        Available priors are beta, zt, dz, C

        Assumes a normal distribution or
        define another distribution from scipy.stats

        Usage
        -----
         add_prior(beta=(p, sigma_p))
         add_prior(beta=scipy.stats.norm(p, sigma_p))
        """

        for key in kwargs:
            if key in self.prior:
                prior = kwargs[key]
                if type(prior) == tuple:
                    p, sigma_p = prior
                    pdf = stats.norm(p, sigma_p)
                elif type(prior) == stats._distn_infrastructure.rv_frozen:
                    pdf = prior
                else:
                    raise ValueError("Use a distribution from scipy.stats module")

                # add prior PDF to dictionary
                self.prior_pdf[key] = pdf
                self.prior[key] = list(pdf.args)

            else:
                raise ValueError("prior must be one of {}".format(self.prior.keys()))


    def reset_priors(self):
        self.prior = {'beta':None, 'zt':None, 'dz':None, 'C':None}
        self.prior_pdf = {'beta':None, 'zt':None, 'dz':None, 'C':None}


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
                prior_args = self.prior[key]
                if prior_args is not None:
                    c += self.objective_function(val, *prior_args)
        return c

    def objective_function(self, x, x0, sigma_x0, *args):
        """
        Objective function used in objective_routine

        Parameters
        ---------
         x        : float, ndarray
         x0       : float, ndarray
         sigma_x0 : float, ndarray

        Returns
        -------
         misfit   : float
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
         when this occurs we set the misfit to a very large number
        """
        beta, zt, dz, C = x
        with warnings.catch_warnings() as w:
            warnings.simplefilter("ignore")
            Phi_syn = bouligand2009(beta, zt, dz, kh, C)

        misfit = self.objective_function(Phi_syn, Phi_exp, 1.0)
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
         window  : float - size of window in metres
         xc      : float - centroid x values
         yc      : float - centroid y values
         beta    : float - fractal parameter (starting value)
         zt      : float - top of magnetic layer (starting value)
         dz      : float - thickness of magnetic layer (starting value)
         C       : float - field constant (starting value)

        Returns
        -------
         beta    : float - fractal parameters
         zt      : float - top of magnetic layer
         dz      : float - thickness of magnetic layer
         C       : float - field constant
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
        
        Parameters
        ----------
         window  : float - size of window in metres
         xc_list : ndarray shape(l,) - centroid x values 
         yc_list : ndarray shape(l,) - centroid y values 
         beta    : float - fractal parameter 
         zt      : float - top of magnetic layer
         dz      : float - thickness of magnetic layer
         C       : float - field constant

        Returns
        -------
         beta    : ndarray shape(l,) - fractal parameters
         zt      : ndarray shape(l,) - top of magnetic layer
         dz      : ndarray shape(l,) - thickness of magnetic layer
         C       : ndarray shape(l,) - field constant

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


    def sensitivity_routine(self, nsim, window, xc_list, yc_list, beta=3.0, zt=1.0, dz=20.0, C=5.0, taper=np.hanning, process_subgrid=None, **kwargs):
        """
        Iterate through a list of centroids to compute the mean and
        standard deviation of beta, zt, dz, C by perturbing their
        prior distributions (if provided by the user - see add_prior).
        
        Parameters
        ----------
         nsim    : int - number of Monte Carlo simulations
         window  : float - size of window in metres
         xc_list : ndarray shape(l,) - centroid x values 
         yc_list : ndarray shape(l,) - centroid y values 
         beta    : float - starting fractal parameter 
         zt      : float - starting top of magnetic layer
         dz      : float - starting thickness of magnetic layer
         C       : float - starting field constant

        Returns
        -------
         beta    : ndarray shape(l,2) - fractal parameters (mean, stdev)
         zt      : ndarray shape(l,2) - top of magnetic layer (mean, stdev)
         dz      : ndarray shape(l,2) - thickness of magnetic layer (mean, stdev)
         C       : ndarray shape(l,2) - field constant (mean, stdev)

        """

        n = len(xc_list)
        
        if n != len(yc_list):
            raise ValueError("xc_list and yc_list must be the same size")


        # storage vectors for centroids (mean, std)
        Bopt  = np.empty((n,2))
        ztopt = np.empty((n,2))
        dzopt = np.empty((n,2))
        Copt  = np.empty((n,2))

        # storage vectors for nsim
        Bsim  = np.empty(nsim)
        ztsim = np.empty(nsim)
        dzsim = np.empty(nsim)
        Csim  = np.empty(nsim)


        use_keys = []
        for key in self.prior_pdf:
            prior_pdf = self.prior_pdf[key]
            if prior_pdf is not None:
                use_keys.append(key)


        for i in range(0, n):
            xc = xc_list[i]
            yc = yc_list[i]

            for sim in range(0, nsim):

                # randomly generate new prior values within PDF
                for key in use_keys:
                    prior_pdf = self.prior_pdf[key]
                    self.prior[key][0] = prior_pdf.rvs()


                # run the optimisation routine
                Bsim[sim], ztsim[sim], dzsim[sim], Csim[sim] = self.optimise(window, xc, yc,\
                                                                             beta=beta, zt=zt, dz=dz, C=C,\
                                                                             taper=np.hanning, process_subgrid=None, **kwargs)

            # store mean, stdev pairs
            Bopt[i] = np.mean(Bsim), np.std(Bsim)
            ztopt[i] = np.mean(ztsim), np.std(ztsim)
            dzopt[i] = np.mean(dzsim), np.std(dzsim)
            Copt[i] = np.mean(Csim), np.std(Csim)


        # restore priors
        for key in use_keys:
            prior_pdf = self.prior_pdf[key]
            self.prior[key] = list(prior_pdf.args)

        return Bopt, ztopt, dzopt, Copt
