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
from sparse.linalg import spsolve

try: range=xrange
except: pass


class CurieTherm(object):

    def __init__(self, minZ, maxZ, resZ, **kwargs):

        self.npoints = resZ
        self.extent = (minZ, maxZ)
        self.coords = np.linspace(minZ, maxZ, resZ)
        self.nodes = np.arange(0, resZ, dtype=np.int)
        self.sizes = (resZ, resZ)

        width = kwargs.pop('stencil_width', 1)
        self.stencil_width = 2*width + 1

        closure = [(0,-2),(2,0),(1,-1)]
        self.closure = self._create_closure_object(closure)


        self._initialise_COO_vectors()
        self._initialise_boundary_dictionary()


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
        self.diffusivity = diffusivity
        self.heat_sources = heat_sources
        return


    def boundary_condition(self, wall, val, flux=False):
        """
        Set the boundary conditions on each wall of the domain.
        By default each wall is a Neumann (flux) condition.
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
        Construct the coefficient matrix
        i.e. matrix A in Ax = b

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

        u = self.diffusivity

        k = np.zeros(nn + 2)
        k[1:-1] = u

        for i in range(0, self.stencil_width):
            obj = self.closure[i]

            rows[i] = nodes
            cols[i] = index[obj].ravel()

            distance = np.linalg.norm(self.coords[cols[i]] - self.coords, axis=1)
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
        i.e. vector b in Ax = b

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
        Construct the matrix A and vector b in Ax = b
        and solve for x
        """
        if matrix is None:
            matrix = self.construct_matrix()
        if rhs is None:
            rhs = self.construct_rhs()
        # res = self.temperature

        T = spsolve(matrix, rhs)
        self.temperature = T

        return T


    def gradient(self, vector, **kwargs):

        return np.gradient(vector, self.coords, **kwargs)


    def heatflux(self):

        T = self.temperature
        k = self.diffusivity
        divT = self.gradient(T)
        return -k*divT