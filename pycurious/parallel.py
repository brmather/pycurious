# Copyright 2018-2019 Ben Mather, Robert Delhaye
#
# This file is part of PyCurious.
#
# PyCurious is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# PyCurious is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PyCurious.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from multiprocessing import Pool, Process, Queue, cpu_count

try:
    range = xrange
except:
    pass


class CurieParallel(object):

    def __init__(self, **kwargs):

        self.max_processors = kwargs.pop("max_processors", cpu_count())


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
                additional arguments to pass to `func`
            kwargs : keyword arguments
                additional keyword arguments to pass to `func`

        Returns:
            out : list of lists
                (depends on output of `func` - see notes)

        Usage:
            An obvious use case is to compute the Curie depth for many
            centroids in parallel.

            >>> self.parallelise_routine(window, xc_list, yc_list, self.optimise)
        
            Each centroid is assigned a new process and sent to a free processor
            to compute. In this case, the output is separate lists of shape(l,)
            for \\( \\beta, z_t, \\Delta z, C \\). If `len(xc_list)=2` then,

            >>> self.parallelise_routine(window, [x1,x2], [y1, y2], self.optimise)
            [[beta1  beta2], [zt1  zt2], [dz1  dz2], [C1  C2]]

            Another example is to parallelise the sensitivity analysis:

            >>> self.parallelise_routine(window, xc_list, yc_list, self.sensitivity, nsim)

            This time the output will be a list of lists for \\( \\beta, z_t, \\Delta z, C \\)
            i.e. if `len(xc_list)=2` is the number of centroids and `nsim=4` is the number of
            simulations then separatee lists will be returned for \\( \\beta, z_t, \\Delta z, C \\).

            >>> self.parallelise_routine(window, [x1,x2], [y1,y2], self.sensitivity, 4)

            which would return:

            ```python
            [[[ beta1a , beta1b , beta1c , beta1d ],   # centroid 1 (x1,y1)
              [ beta2a , beta2b , beta2c , beta2d ]],  # centroid 2 (x2,y2)
             [[   zt1a ,   zt1b ,   zt1c ,   zt1d ],   # centroid 1 (x1,y1)
              [   zt2a ,   zt2b ,   zt2c ,   zt2d ]],  # centroid 2 (x2,y2)
             [[   dz1a ,   dz1b ,   dz1c ,   dz1d ],   # centroid 1 (x1,y1)
              [   dz2a ,   dz2b ,   dz2c ,   dz2d ]]   # centroid 2 (x2,y2)
             [[    C1a ,    C1b ,    C1c ,    C1d ],   # centroid 1 (x1,y1)
              [    C2a ,    C2b ,    C2c ,    C2d ]]]  # centroid 2 (x2,y2)
            ```
        """

        n = len(xc_list)
        if n != len(yc_list):
            raise ValueError("xc_list and yc_list must be the same size")

        xOpt = [[] for i in range(n)]
        processes = []
        q_in = Queue(1)
        q_out = Queue()

        nprocs = self.max_processors

        if nprocs == 1:
            # skip all the OpenMP cruft
            for i in range(n):
                xc = xc_list[i]
                yc = yc_list[i]

                res = func(window, xc, yc, *args, **kwargs)
                xOpt[i] = res

        elif nprocs > 1:
            # more than one processor
            for i in range(nprocs):
                pass_args = [func, q_in, q_out, window]
                pass_args.extend(args)

                p = Process(target=self._func_queue, args=tuple(pass_args), kwargs=kwargs)

                processes.append(p)

            for p in processes:
                p.daemon = True
                p.start()

            # put items in the queue
            sent = [q_in.put((i, xc_list[i], yc_list[i])) for i in range(n)]
            [q_in.put((None, None, None)) for _ in range(nprocs)]

            # get the results
            for i in range(len(sent)):
                index, res = q_out.get()
                xOpt[index] = res

            # wait until each processor has finished
            [p.join() for p in processes]

        else:
            raise ValueError("{} processors is invalid, specify a positive integer value".format(nprocs))

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
