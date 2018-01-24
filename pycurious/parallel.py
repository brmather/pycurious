# -*- coding: utf-8 -*-
import numpy as np
import warnings
from .optimise import CurieOptimise

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
    """

    def __init__(self, grid, xmin, xmax, ymin, ymax, max_window):

        from petsc4py import PETSc
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        self.comm = comm

        # super(CurieParallel, self).__init__(grid, xmin, xmax, ymin, ymax)

        # determine stencil width (should be window size/2)
        ny, nx = grid.shape
        dx = (xmax - xmin)/nx
        dy = (ymax - ymin)/ny
        nw = int(round(max_window)/dx)
        n2w = nw//2

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

        super(CurieParallel, self).subgrid(xc, yc, window)

    
    def sync(self, vector):
        """
        Synchronise a local vector across all processors
        """
        self.lvec.setArray(vector)
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.dm.globalToLocal(self.gvec, self.lvec)
        return self.lvec.array.copy()