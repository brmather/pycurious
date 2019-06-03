"""
Copyright 2018 Ben Mather, Robert Delhaye

This file is part of PyCurious.

PyCurious is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

PyCurious is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with PyCurious.  If not, see <http://www.gnu.org/licenses/>.
"""

# -*- coding: utf-8 -*-
from .grid import CurieGrid, bouligand2009
import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.special import polygamma
from scipy import stats
from multiprocessing import Pool, Process, Queue, cpu_count

try: range=xrange
except: pass


class CurieOptimise(CurieGrid):
    """
    Extends the CurieGrid class to include optimisation routines
    see scipy.optimize.minimize for a description of the algorithm

    Args:
        grid : 2D numpy array
            2D array of magnetic data
        xmin : float
            minimum x bound in metres
        xmax : float
            maximum x bound in metres
        ymin : float
            minimum y bound in metres
        ymax : float
            maximum y bound in metres
    
    Attributes:
        bounds : list of tuples
            lower and upper bounds for beta, zt, dz, C
        prior  : dict
            dictionary of priors for beta, zt, dz, C
    """
    def __init__(self, grid, xmin, xmax, ymin, ymax, **kwargs):

        super(CurieOptimise, self).__init__(grid, xmin, xmax, ymin, ymax)

        # initialise prior dictionary
        self.reset_priors()

        # lower / upper bounds
        # [beta, zt, dz, C]
        lb = [0.0, 0.0, 0.0, None]
        ub = [None]*len(lb)
        bounds = list(zip(lb,ub))
        self.bounds = bounds

        self.max_processors = kwargs.pop("max_processors", cpu_count())

        return

    def add_prior(self, **kwargs):
        """
        Add a prior to the dictionary (tuple)
        Available priors are beta, zt, dz, C

        Assumes a normal distribution or
        define another distribution from scipy.stats

        Usage:
            >>> add_prior(beta=(p, sigma_p))

            >>> add_prior(beta=scipy.stats.norm(p, sigma_p))
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
        """
        Reset priors to uniform distribution
        """
        self.prior = {'beta':None, 'zt':None, 'dz':None, 'C':None}
        self.prior_pdf = {'beta':None, 'zt':None, 'dz':None, 'C':None}


    def objective_routine(self, **kwargs):
        """
        Evaluate the objective routine to find the misfit with priors
        Only keys stored in self.prior will be added to the total misfit

        Usage:
            >>> objective_routine(beta=2.5)

        Returns:
            misfit  : float
                misfit integrated over all observations and priors
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
        Objective function used in `objective_routine`
        Evaluates the l2-norm misfit

        Args:
            x : float, ndarray
            x0 : float, ndarray
            sigma_x0 : float, ndarray

        Returns:
            misfit   : float
        """
        return 0.5*np.sum((x - x0)**2/sigma_x0**2)


    def min_func(self, x, kh, Phi, sigma_Phi):
        """
        Function to minimise

        Args:
            x : array shape (n,)
                array of variables [beta, zt, dz, C]
            kh : array shape (n,)
                wavenumbers (rad/km)
            Phi : array shape (n,)
                radial power spectrum
            sigma_Phi : array shape (n,)
                standard deviation of Phi

        Returns:
            misfit    : float
                sum of misfit (float)

        Notes:
            We purposely ignore all warnings raised by the bouligand2009
            function because some combinations of input parameters will
            trigger an out-of-range warning that will crash the minimiser.
            Instead, the misfit is set to a very large number when this occurs.
        """
        beta, zt, dz, C = x
        with warnings.catch_warnings() as w:
            warnings.simplefilter("ignore")
            Phi_syn = bouligand2009(kh, beta, zt, dz, C)

        misfit = self.objective_function(Phi_syn, Phi, 1.0)
        if not np.isfinite(misfit):
            misfit = 1e99
        else:
            misfit += self.objective_routine(beta=beta, zt=zt, dz=dz, C=C)
        return misfit


    def optimise(self, window, xc, yc, beta=3.0, zt=1.0, dz=10.0, C=5.0, taper=np.hanning, process_subgrid=None, **kwargs):
        """
        Find the optimal parameters of beta, zt, dz, C for a given
        centroid (xc,yc) and window size.

        Args:
            window  : float
                size of window in metres
            xc : float
                centroid x values
            yc : float
                centroid y values
            beta : float
                fractal parameter (starting value)
            zt : float
                top of magnetic layer (starting value)
            dz : float
                thickness of magnetic layer (starting value)
            C : float
                field constant (starting value)
            taper : taper (default=np.hanning)
                taper function, set to None for no taper function
            process_subgrids : function
                a custom function to process the subgrid
            kwargs : keyword arguments
                to pass to radial_spectrum.

        Returns:
            beta : float
                fractal parameters
            zt : float
                top of magnetic layer
            dz : float
                thickness of magnetic layer
            C : float
                field constant
        """

        if type(process_subgrid) == type(None):
            # dummy function
            def process_subgrid(subgrid):
                return subgrid

        # initial constants for minimisation
        # w = 1.0 # weight low frequency?

        x0 = np.array([beta, zt, dz, C])

        # get subgrid
        subgrid = self.subgrid(window, xc, yc)
        subgrid = process_subgrid(subgrid)

        # compute radial spectrum
        k, Phi, sigma_Phi = self.radial_spectrum(subgrid, taper=taper, **kwargs)

        # minimise function
        res = minimize(self.min_func, x0, args=(k, Phi, sigma_Phi), bounds=self.bounds)
        return res.x


    def _func_queue(self, func, q_in, q_out, window, *args, **kwargs):
        """ Retrive processes from the queue """
        while True:
            pos, xc, yc = q_in.get()
            if pos is None:
                break

            pass_args = [window, xc, yc]
            pass_args.extend(args)

            res = func(*pass_args, **kwargs)
            q_out.put((pos, res))
        return


    def parallelise_routine(self, window, xc_list, yc_list, func, *args, **kwargs):
        """
        Implements shared memory multiprocessing to split multiple
        evaluations of a function centroids across processors.

        Supply the window size and lists of x,y coordinates to a function
        along with any additional arguments or keyword arguments.

        Args:
         window : float
            size of window in metres
         xc_list : array shape (l,)
            centroid x values
         yc_list : array shape (l,)
            centroid y values
         func : function
            Python function to evaluate in parallel
         args : arguments
            additional arguments to pass to func
         kwargs : keyword arguments
            additional keyword arguments to pass to func

        Returns:
            out : list
                (depends on output of func - see notes)

        Usage:
            An obvious use case is to compute the Curie depth for many
            centroids in parallel.

            >>> self.parallelise_routine(window, xc_list, yc_list, self.optimise)
        
            Each centroid is assigned a new process and sent to a free processor
            to compute. In this case, the output is separate lists of shape(l,)
            for beta, zt, dz, and C.

            Another example is to parallelise the sensitivity analysis:

            >>> self.parallelise_routine(window, xc_list, yc_list, self.sensitivity, nsim)

            This time the output will be a list of lists for beta, zt, dz, and C
            i.e. if nc=2 is the number of centroids and nsim=4 is the number of
            simulations then separatee lists [[k1, k2, k3, k4], [k1, k2, k3, k4]]
            will be returned for beta, zt, dz, and C.
        """

        n = len(xc_list)
        if n != len(yc_list):
            raise ValueError("xc_list and yc_list must be the same size")

        xOpt = [[] for i in range(n)]
        processes = []
        q_in = Queue(1)
        q_out = Queue()

        nprocs = self.max_processors

        for i in range(nprocs):
            pass_args = [func, q_in, q_out, window]
            pass_args.extend(args)

            p = Process(target=self._func_queue,\
                        args=tuple(pass_args),\
                        kwargs=kwargs)

            processes.append(p)

        for p in processes:
            p.daemon = True
            p.start()

        # put items in the queue
        sent = [q_in.put((i, xc_list[i], yc_list[i])) for i in range(n)]
        [q_in.put((None, None, None)) for _ in range(nprocs)]

        # get the results
        for i in range(len(sent)):
            i, res = q_out.get()
            xOpt[i] = res


        # wait until each processor has finished
        [p.join() for p in processes]

        # process dimensions of output
        ndim = np.array(res).ndim

        if ndim == 1:
            # return separate lists of beta, zt, dz, C
            xOpt = np.vstack(xOpt)
            return list(xOpt.T)
        elif ndim > 1:
            # return lists of beta, zt, dz, C for each centroid
            xOpt = np.hstack(xOpt)
            out = list(xOpt)
            for i in range(len(out)):
                out[i] = np.split(out[i], n)
            return out
        else:
            raise ValueError("Cannot determine shape of output")



    def optimise_routine(self, window, xc_list, yc_list, beta=3.0, zt=1.0, dz=10.0, C=5.0, taper=np.hanning, process_subgrid=None, **kwargs):
        """
        Iterate through a list of centroids to compute the optimal values
        of beta, zt, dz, C for a given window size.
        
        Args:
            window : float
                size of window in metres
            xc_list : ndarray shape (l,)
                centroid x values 
            yc_list : ndarray shape (l,)
                centroid y values 
            beta : float
                fractal parameter 
            zt : float
                top of magnetic layer
            dz : float
                thickness of magnetic layer
            C : float
                field constant
            taper : function
                taper function (default=np.hanning)
                set to None for no taper function
         process_subgrids : func
            a custom function to process the subgrid
         kwargs : keyword arguments
            to pass to radial_spectrum.

        Returns:
            beta : ndarray shape (l,)
                fractal parameters
            zt : ndarray shape (l,)
                top of magnetic layer
            dz : ndarray shape (l,)
                thickness of magnetic layer
            C : ndarray shape (l,)
                field constant

        """
        return self.parallelise_routine(window, xc_list, yc_list, self.optimise, beta, zt, dz, C, taper, process_subgrid, **kwargs)


    def metropolis_hastings(self, window, xc, yc, nsim, burnin, x_scale=None, beta=3.0, zt=1.0, dz=10.0, C=5.0, taper=np.hanning, process_subgrid=None, **kwargs):
        """
        MCMC algorithm using a Metropolis-Hastings sampler.

        Evaluates a Markov-Chain for starting values of beta, zt, dz, C
        and returns the ensemble of model realisations.
        
        Args:
            window : float
                size of window in metres
            xc : float
                centroid x values
            yc : float
                centroid y values
            nsim : int
                number of simulations
            burnin : int
                number of burn-in simulations before to nsim
            x_scale: float(4) (optional)
                scaling factor for new proposals
                default is [1,1,1,1] for [beta, zt, dz, C] - see notes
            beta : float
                fractal parameter (starting value)
            zt : float
                top of magnetic layer (starting value)
            dz : float
                thickness of magnetic layer (starting value)
            C : float
                field constant (starting value)

        Returns:
            beta : ndarray shape (nsim,)
                fractal parameter
            zt : ndarray shape (nsim,)
                top of magnetic layer
            dz : ndarray shape (nsim,)
                thickness of magnetic layer
            C : ndarray shape (nsim,)
                field constant

        Notes:
            nsim, burnin, and x_scale should be tweaked for optimal performance
            Use starting values of beta, zt, dz, C relatively close to the solution
            - C can easily found from the mean of the radial power spectrum.

            During the burn-in stage we apply tempering to the PDF to iterate closer
            towards the solution. This has the effect of smoothing out the posterior
            so that minima can be more easily found. This is necessary here because
            large portions of the posterior probability are zero.
            see see Sambridge 2013, DOI:10.1093/gji/ggt342 for more information.
        """
        if type(process_subgrid) == type(None):
            # dummy function
            def process_subgrid(subgrid):
                return subgrid


        samples = np.empty((nsim, 4))
        x0 = np.array([beta, zt, dz, C])

        if type(x_scale) == type(None):
            x_scale = np.ones(4)


        # get subgrid
        subgrid = self.subgrid(window, xc, yc)
        subgrid = process_subgrid(subgrid)

        # compute radial spectrum
        k, Phi, sigma_Phi = self.radial_spectrum(subgrid, taper=taper, **kwargs)

        P0 = np.exp(-self.min_func(x0, k, Phi, sigma_Phi)/1000)

        # Burn-in phase
        for i in range(burnin):
            # add random perturbation
            x1 = x0 + np.random.normal(size=4)*x_scale

            # evaluate proposal probability + tempering
            P1 = np.exp(-self.min_func(x1, k, Phi, sigma_Phi)/1000)

            # iterate towards MAP estimate
            if P1 > P0:
                x0 = x1
                P0 = P1

        P0 = np.exp(-self.min_func(x0, k, Phi, sigma_Phi))

        # Now sample posterior
        for i in range(nsim):
            # add random perturbation
            x1 = x0 + np.random.normal(size=4)*x_scale

            # evaluate proposal probability
            P0 = max(P0, 1e-99)
            P1 = np.exp(-self.min_func(x1, k, Phi, sigma_Phi))

            P = min(P1/P0, 1.0)

            # randomly accept probability
            if np.random.rand() <= P:
                x0 = x1
                P0 = P1

            samples[i] = x0

        return list(samples.T)



    def sensitivity(self, window, xc, yc, nsim, beta=3.0, zt=1.0, dz=10.0, C=5.0, taper=np.hanning, process_subgrid=None, **kwargs):
        """
        Iterate through a list of centroids to compute the mean and
        standard deviation of beta, zt, dz, C by perturbing their
        prior distributions (if provided by the user - see add_prior).
        
        Args:
            nsim : int
                number of Monte Carlo simulations
            window : float
                size of window in metres
            xc : float
                centroid x values
            yc : float
                centroid y values
            nsim : int
                - number of simulations
            beta : float
                starting fractal parameter 
            zt : float
                starting top of magnetic layer
            dz : float
                starting thickness of magnetic layer
            C : float
                starting field constant


        Returns:
            beta : ndarray shape (nsim,)
                fractal parameters
            zt : ndarray shape (nsim,)
                top of magnetic layer
            dz : ndarray shape (nsim,)
                thickness of magnetic layer
            C : ndarray shape (nsim,)
                field constant
        """
        if type(process_subgrid) == type(None):
            # dummy function
            def process_subgrid(subgrid):
                return subgrid


        samples = np.empty((nsim, 4))
        x0 = np.array([beta, zt, dz, C])
        
        use_keys = []
        for key in self.prior_pdf:
            prior_pdf = self.prior_pdf[key]
            if prior_pdf is not None:
                use_keys.append(key)
        
        # get subgrid
        subgrid = self.subgrid(window, xc, yc)
        subgrid = process_subgrid(subgrid)

        # compute radial spectrum
        k, Phi, sigma_Phi = self.radial_spectrum(subgrid, taper=taper, **kwargs)

        for sim in range(0, nsim):
            # randomly generate new prior values within PDF
            for key in use_keys:
                prior_pdf = self.prior_pdf[key]
                self.prior[key][0] = prior_pdf.rvs()

            # minimise function
            rPhi = np.random.normal(Phi, sigma_Phi)
            res = minimize(self.min_func, x0, args=(k, rPhi, sigma_Phi), bounds=self.bounds)
            samples[sim] = res.x


        # restore priors
        for key in use_keys:
            prior_pdf = self.prior_pdf[key]
            self.prior[key] = list(prior_pdf.args)

        return list(samples.T)
