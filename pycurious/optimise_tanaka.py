# -*- coding: utf-8 -*-
from .grid import CurieGrid
import numpy as np
import warnings
from scipy.optimize import curve_fit
from multiprocessing import Pool, Process, Queue, cpu_count

try:
    range = xrange
except:
    pass


class CurieOptimiseTanaka(CurieGrid):

    def __init__(self, grid, xmin, xmax, ymin, ymax, **kwargs):

        super(CurieOptimiseTanaka, self).__init__(grid, xmin, xmax, ymin, ymax)

        self.max_processors = kwargs.pop("max_processors", cpu_count())

    def optimise(self, 
        window, 
        xc, 
        yc, 
        z0_range=(0,0.1), 
        zt_range=(0.2,0.3), 
        taper=np.hanning, 
        process_subgrid=None):

        def linear_func(x, a, b):
            return a*x + b

        if process_subgrid is None:
            # dummy function
            def process_subgrid(subgrid):
                return subgrid

        # get subgrid
        subgrid = self.subgrid(window, xc, yc)
        subgrid = process_subgrid(subgrid)

        # calcualte spectra
        k, Phi, sigma_Phi = grid.radial_spectrum(subgrid, taper=taper, power=0.5)
        Phi_n = np.log(np.exp(Phi)/k)
        sigma_Phi_n = np.log(np.exp(sigma_Phi)/k)

        z0_min, z0_max = (0,0.1)
        zt_min, zt_max = (0.2,0.3)

        # divide everything by 2 pi
        k_new = k/(2*np.pi)
        Phi_new = Phi/(2*np.pi)
        Phi_n_new = Phi_n/(2*np.pi)

        sigma_Phi_new = sigma_Phi/(2*np.pi)
        sigma_Phi_n_new = sigma_Phi_n/(2*np.pi)

        # mask zt range
        mask_zt = np.logical_and(k_new >= zt_min, k_new <= zt_max)
        k_zt = k_new[mask_zt]
        Phi_zt = Phi_new[mask_zt]
        sigma_Phi_zt = sigma_Phi_new[mask_zt]

        # mask z0 range
        mask_z0 = np.logical_and(k_new >= z0_min, k_new <= z0_max)
        k_z0 = k_new[mask_z0]
        Phi_z0 = Phi_n_new[mask_z0] # weighted
        sigma_Phi_z0 = sigma_Phi_n_new[mask_z0]

        ## calcualte linear regression and return the residual (sum of squared errors (SSE))
        (zt_slope, zt_intercept), zt_cov = curve_fit(linear_func, k_zt, Phi_zt, sigma=sigma_Phi_zt, absolute_sigma=True)
        (z0_slope, z0_intercept), z0_cov = curve_fit(linear_func, k_z0, Phi_z0, sigma=sigma_Phi_z0, absolute_sigma=True)

        # standard deviation is the square root of the covariance matrix
        zt_slope_stdev = np.sqrt(np.diag(zt_cov))[0]
        z0_slope_stdev = np.sqrt(np.diag(z0_cov))[0]

        return (zt_slope, z0_slope, zt_slope_stdev, z0_slope_stdev)

    def optimise_routine(self,
        window, 
        xc_list, 
        yc_list, 
        z0_range=(0,0.1), 
        zt_range=(0.2,0.3), 
        taper=np.hanning, 
        process_subgrid=None):

        return self.parallelise_routine(
            window,
            xc_list,
            yc_list,
            self.optimise,
            z0_range,
            zt_range,
            taper,
            process_subgrid,
            **kwargs
        )

    def calculate_CPD(self, zt, z0, sigma_zt, sigma_z0):
        """
        Compute the Curie depth from the results of tanaka1999

        Args:
            zt : float / 1D array
                top of the magnetic source
            z0 : float / 1D array
                centroid depth of the magnetic source
            sigma_zt : float / 1D array
                standard deviation of `zt`
            sigma_z0 : float / 1D array
                standard deviation of `z0`

        Returns:
            CPD : float / 1D array
                estimated Curie point depth at bottom of magnetic source
            CPD_stdev : float / 1D array
                standard deviation
        """
        CPD = 2.0*z0 - zt
        CPD_stdev = np.sqrt(sigma_zt**2 + (sigma_z0*2)**2)
        return (CPD, CPD_stdev)
