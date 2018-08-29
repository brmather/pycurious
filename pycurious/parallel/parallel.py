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
import numpy as np
import warnings
from ..optimise import CurieOptimise

try: range=xrange
except: pass


class CurieParallel(CurieOptimise):
    """
    A parallel implementation built on the PETSc DMDA mesh structure
    Almost a trivially parallel problem: each processor works locally
    on their portion of the grid. A stipulation being that the window
    size must not exceed the thickness of shadow zones or else the
    subgrid will be truncated.

    This inherits from CurieOptimise and CurieGrid classes.


    Parameters
    ----------
     grid       : 2D array of magnetic data
     xmin       : minimum x bound in metres
     xmax       : maximum x bound in metres
     ymin       : minimum y bound in metres
     ymax       : maximum y bound in metres
     max_window : maximum size of the windows
                : this will be enforced for all methods
    
    Attributes
    ----------
     bounds     : lower and upper bounds for beta, zt, dz, C
     prior      : dictionary of priors for beta, zt, dz, C
     coords     : local xy coordinates
    """

    def __init__(self, grid, xmin, xmax, ymin, ymax, max_window):

        from petsc4py import PETSc
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        self.comm = comm
        self.MPI = MPI

        # super(CurieParallel, self).__init__(grid, xmin, xmax, ymin, ymax)

        # determine stencil width (should be window size/2)
        ny, nx = grid.shape
        dx = (xmax - xmin)/nx
        dy = (ymax - ymin)/ny
        nw = max_window/dx
        n2w = int(nw/2) + 1 # add some buffer

        if abs(dx - dy) > 1.0:
            warnings.warn("node spacing should be identical {}".format((dx,dy)), RuntimeWarning)


        dm = PETSc.DMDA().create(dim=2, sizes=(nx,ny), stencil_width=n2w, comm=comm)
        dm.setUniformCoordinates(xmin, xmax, ymin, ymax)

        self.dm = dm
        self.lgmap = dm.getLGMap()
        self.lvec = dm.createLocalVector()
        self.gvec = dm.createGlobalVector()

        coords = dm.getCoordinatesLocal().array.reshape(-1, 2)
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        self.coords = coords
        self.max_window = max_window

        (imin, imax), (jmin, jmax) = dm.getGhostRanges()

        # now decompose grid for each processor

        grid_chunk = grid[jmin:jmax, imin:imax]

        super(CurieParallel, self).__init__(grid_chunk, xmin, xmax, ymin, ymax)

        reduce_methods = {'sum': MPI.SUM, 'max': MPI.MAX, 'min': MPI.MIN, 'mean': MPI.SUM}
        self._reduce_methods = reduce_methods


    def subgrid(self, xc, yc, window):
        """
        Extract a subgrid from the data at a window around
        the point (xc,yc)
        
        Parameters
        ----------
         xc      : x coordinate
         yc      : y coordinate
         window  : size of window in metres

        Returns
        -------
         data    : subgrid
        """
        if window > self.max_window:
            raise ValueError("Max window size is {}".format(self.max_window))

        return super(CurieParallel, self).subgrid(window, xc, yc)

    
    def sync(self, vector):
        """
        Synchronise a local vector across all processors
        """
        self.lvec.setArray(vector)
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.dm.globalToLocal(self.gvec, self.lvec)
        return self.lvec.array.copy()


    def add_parallel_prior(self, **kwargs):
        """
        Add a prior to the dictionary (tuple)
        This version broadcasts just the values from the root processor

        Available priors are beta, zt, dz, C

        Usage
        -----
         add_parallel_prior(beta=(p, sigma_p))
        """

        comm = self.comm
        MPI = self.MPI

        for key in kwargs:
            if key in self.prior:
                prior = kwargs[key]
                p  = np.array(prior[0]) # prior
                dp = np.array(prior[1]) # uncertainty

                comm.Bcast([p, MPI.DOUBLE], root=0)
                comm.Bcast([dp,MPI.DOUBLE], root=0)

                self.prior[key] = (float(p), float(dp))
            else:
                raise ValueError("prior must be one of {}".format(self.prior.keys()))


    def distribute_prior(self, method, **kwargs):
        """
        Distribute priors across all processors using a specific method.

        Parameters
        ----------
         method  : operation to reduce the local priors to a single value
                 : choose from one of 'sum', 'mean', 'min', 'max'
         kwargs  : (prior, sigma_prior) tuple

        Notes
        -----
         add_prior will broadcast the prior on the root processor (rank=0)
         distribute_prior enacts a MPI Allreduce operation.
        """

        comm = self.comm
        MPI = self.MPI

        if method in self._reduce_methods:
            op = self._reduce_methods[method]
        else:
            raise ValueError("choose one of the following methods {}".format(self._reduce_methods.keys()))

        for key in kwargs:
            if key in self.prior:
                prior = kwargs[key]
                local_p  = np.array(prior[0]) # prior
                local_dp = np.array(prior[1]) # uncertainty

                global_p = np.array(0.0)
                global_dp = np.array(0.0)

                comm.Allreduce([local_p, MPI.DOUBLE], [global_p, MPI.DOUBLE], op=op)
                comm.Allreduce([local_dp, MPI.DOUBLE], [global_dp, MPI.DOUBLE], op=op)
                if method == 'mean':
                    global_p  /= comm.size
                    global_dp /= comm.size

                self.prior[key] = (float(global_p), float(global_dp))
            else:
                raise ValueError("prior must be one of {}".format(self.prior.keys()))