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

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from multiprocessing import Pool, Process, Queue, cpu_count

try: range=xrange
except: pass


class CurieTherm(object):
    """
    Implicit 1d solver for the steady-state heat equation
    over a structured grid using sparse matrices.

    Parameters
    ----------
     minZ : float, minimum Cartesian coordinate
     maxZ : float, maximum Cartesian coordinate
     resZ : int, resolution in the z direction
     kwargs : dict, keyword arguments to pass to control
       optional functionality e.g.
       - stencil_width=int : number of node neighbours to
           include in each matrix row

    Attributes
    ----------
     diffusivity  : float shape(n,) thermal conductivity field
     heat_sources : float shape(n,) heat source field
     temperature  : float shape(n,) temperature field

     npoints      : int, number of nodes in the mesh
     sizes        : (int,int) dimensions of the sparse matrix
     
    """

    def __init__(self, minZ, maxZ, resZ, **kwargs):

        self.npoints = resZ
        self.extent = (minZ, maxZ)
        self.coords = np.linspace(minZ, maxZ, resZ)
        self.nodes = np.arange(0, resZ, dtype=np.int)
        self.sizes = (resZ, resZ)

        width = kwargs.pop('stencil_width', 1)
        self.stencil_width = 2*width + 1

        closure = [(0,-2), (2,0), (1,-1)]
        self.closure = self._create_closure_object(closure)


        self._initialise_COO_vectors()
        self._initialise_boundary_dictionary()
        self.diffusivity  = np.zeros(resZ)
        self.heat_sources = np.zeros(resZ)
        self.temperature  = np.zeros(resZ)


    def _initialise_COO_vectors(self):

        nn = self.npoints

        index = np.empty(nn + 2, dtype=np.int)
        index.fill(-1)
        self.index = index

        self.rows = np.empty((self.stencil_width, nn), dtype=np.int)
        self.cols = np.empty((self.stencil_width, nn), dtype=np.int)
        self.vals = np.empty((self.stencil_width, nn))


    def _initialise_boundary_dictionary(self):

        nn = self.npoints
        coords = self.coords
        minZ, maxZ = self.extent

        # Setup boundary dictionary
        bc = dict()

        m0 = coords == minZ
        m1 = coords == maxZ
        d0 = abs(coords[1] - coords[0])
        d1 = abs(coords[-1] - coords[-2])

        bc["minZ"] = {"val": 0.0, "delta": d0, "flux": True, "mask": m0}
        bc["maxZ"] = {"val": 0.0, "delta": d1, "flux": True, "mask": m1}

        self.bc = bc
        self.dirichlet_mask = np.zeros(nn, dtype=bool)


    def _create_closure_object(self, closure):

        nn = self.npoints
        obj = [[0] for i in range(self.stencil_width)]

        for i in range(0, self.stencil_width):
            # construct slicing object
            start, end = closure[i-j]
            obj[i] = slice(start, nn+end+2)

        return obj


    def update_properties(self, diffusivity, heat_sources):
        """
        Update diffusivity and heat sources
        """
        self.diffusivity[:] = diffusivity
        self.heat_sources[:] = heat_sources
        return


    def boundary_condition(self, wall, val, flux=False):
        """
        Set the boundary conditions on each wall of the domain.
        By default each wall is a Dirichlet condition.

        Parameters
        ----------
         wall : str, wall to assign bc - 'minZ' or 'maxZ'
         val  : float or array(n,) value(s) to assign to wall
         flux : bool, toggle type of boundary condition
           True = Neumann flux boundary condition
           False = Dirichlet boundary condition (default)

        Notes
        -----
         If flux=True, positive val indicates a flux vector towards the centre
         of the domain.

         val can be a vector with the same number of elements as the wall
        """
        wall = str(wall)

        if wall in self.bc:
            self.bc[wall]["val"]  = np.array(val, copy=True)
            self.bc[wall]["flux"] = flux
            d = self.bc[wall]

            mask = d['mask']

            if flux:
                self.dirichlet_mask[mask] = False
                self.bc[wall]["val"] /= -d['delta']
            else:
                self.dirichlet_mask[mask] = True

        else:
            raise ValueError("Wall should be one of {}".format(self.bc.keys()))


    def construct_matrix(self):
        """
        Construct a sparse coefficient matrix
        i.e. matrix A in AT = b

        Notes
        -----
         We vectorise the 7-point stencil for fast matrix insertion.
         An extra border of dummy values around the domain allows for automatic
         Neumann (flux) boundary creation.
         These are stomped on if there are any Dirichlet conditions.
        """

        nodes = self.nodes
        nn = self.npoints

        index = self.index

        rows = self.rows
        cols = self.cols
        vals = self.vals

        dirichlet_mask = self.dirichlet_mask
        coords = self.coords.reshape(-1,1)

        u = self.diffusivity

        k = np.zeros(nn + 2)
        k[1:-1] = u

        for i in range(0, self.stencil_width):
            obj = self.closure[i]

            rows[i] = nodes
            cols[i] = index[obj].ravel()

            distance = np.linalg.norm(coords[cols[i]] - coords, axis=1)
            distance[distance==0] = 1e-12 # protect against dividing by zero
            delta = 1.0/(2.0*distance**2)

            vals[i] = delta*(k[obj] + u).ravel()


        # Dirichlet boundary conditions (duplicates are summed)
        cols[:,dirichlet_mask] = nodes[dirichlet_mask]
        vals[:,dirichlet_mask] = 0.0

        # zero off-grid coordinates
        vals[cols < 0] = 0.0

        # centre point
        vals[-1] = 0.0
        vals[-1][dirichlet_mask] = -1.0


        row = rows.ravel()
        col = cols.ravel()
        val = vals.ravel()


        # mask off-grid entries and sum duplicates
        mask = col >= 0
        row = row[mask]
        col = col[mask]
        val = val[mask]

        mat = sparse.coo_matrix((val, (row, col)), shape=self.sizes).tocsr()
        mat.sum_duplicates()
        diag = np.ravel(mat.sum(axis=1))
        diag *= -1
        mat.setdiag(diag)

        return mat


    def construct_rhs(self):
        """
        Construct the right-hand-side vector
        i.e. vector b in AT = b

        Notes
        -----
         Boundary conditions are grabbed from the dictionary and
         summed to the rhs.
         Be careful of duplicate entries on the corners!!
        """
        
        rhs = -1.0*self.heat_sources

        for wall in self.bc:
            val  = self.bc[wall]['val']
            flux = self.bc[wall]['flux']
            mask = self.bc[wall]['mask']
            if flux:
                rhs[mask] += val
            else:
                rhs[mask] = val

        return rhs


    def solve(self, matrix=None, rhs=None):
        """
        Construct the matrix A and vector b in AT = b and solve for T
        (i.e. temperature field)

        Arguments
        ---------
         matrix : (optional) scipy sparse matrix object
                 build using construct_matrix()
         rhs    : (optional) numpy right-hand-side vector
                 build using construct_rhs()

        Returns
        -------
         sol    : 1d array shape(n,) temperature solution

        Notes
        -----
         The solution to the system of linear equations AT = b is solved
         with spsolve in the scipy.sparse.linalg module
        """
        if matrix is None:
            matrix = self.construct_matrix()
        if rhs is None:
            rhs = self.construct_rhs()
        # res = self.temperature

        T = spsolve(matrix, rhs)
        self.temperature[:] = T

        return T


    def gradient(self, vector, **kwargs):
        """
        Calculate gradient of a vector
        
        Arguments
        ---------
         vector : 1d array of shape(n,)

        Returns
        -------
         dvdz   : 1d array of shape(n,)
                 derivative of vector in z direction
        """
        return np.gradient(vector, self.coords, **kwargs)


    def heatflux(self):
        """
        Calculate the heat flux from the conductivity field
        and temperature solution

        Returns
        -------
         qz  : 1d array shape(n,) heat flux vector
        """

        T = self.temperature
        k = self.diffusivity
        divT = self.gradient(T)
        return -k*divT


    def isosurface(self, vector, isoval, interp='linear'):
        """
        Calculate an isosurface along the z axis

        Parameters
        ----------
         vector : array, the same size as the mesh (n,)
         isoval : float, isosurface value
         interp : str, interpolation method can be either
            'nearest' - nearest neighbour interpolation
            'linear'  - linear interpolation
        
        Returns
        -------
         z_interp : isosurface
        """
        coords = self.coords
        sort_idx = np.abs(vector - isoval).argsort()    
        i0 = sort_idx[0]
        z0 = coords[i0]

        if interp == 'linear':
            v0 = vector[i0]
            
            # identify next nearest node
            i1 = sort_idx[1]
            z1 = coords[i1]
            v1 = vector[i1]

            vmin = min(v0,v1)
            vmax = max(v0,v1)

            ratio = np.array([isoval, vmin, vmax])
            ratio -= ratio.min()
            ratio /= ratio.max()
            z_interp = ratio[0]*z1 + (1.0 - ratio[0])*z0
            return z_interp
        elif interp == 'nearest':
            return z0
        else:
            raise ValueError("enter a valid interp method: 'linear' or 'nearest'")
