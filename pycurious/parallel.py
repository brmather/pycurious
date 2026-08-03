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

"""
Shared-memory parallelism for evaluating a routine over many centroids.

`CurieParallel` is a mixin inherited by `pycurious.grid.CurieGrid`, and hence
by both optimisation classes. It distributes centroids across processors and
reassembles the results.

## Running in parallel from a script

On macOS and Windows the default start method is *spawn*, which re-imports the
calling module in every worker. A script that calls `parallelise_routine` at
module level must guard its entry point, or each worker will re-execute the
whole script:

```python
if __name__ == "__main__":
    grid.optimise_routine(window, xc_list, yc_list, ...)
```

Jupyter notebooks are unaffected.

Spawn also requires every argument to be picklable. A `taper` or
`process_subgrid` function defined in a notebook cell or in `__main__` cannot
be pickled; `parallelise_routine` detects this before starting any workers and
falls back to serial evaluation with a warning rather than deadlocking.
"""

import pickle
import queue as _queue
import warnings
from multiprocessing import Process, Queue, cpu_count, get_start_method

import numpy as np


def stochastic(func):
    """
    Mark a routine as one that `CurieParallel.parallelise_routine` should seed.

    Routines that draw random numbers take a `seed` keyword and carry this
    marker; deterministic ones do neither, and are left alone. Keeping the
    distinction here rather than in each routine's signature means a
    deterministic routine needs no `seed` parameter to absorb and ignore, and
    that adding a new one cannot silently produce an unseeded map.

    Signature inspection would not do instead: every routine here accepts
    `**kwargs`, so it cannot distinguish one that wants a seed from one that
    would forward it to the taper and fail there.
    """
    func.wants_seed = True
    return func


class CurieParallel(object):
    def __init__(self, **kwargs):

        self.max_processors = kwargs.pop("max_processors", cpu_count())

    def _func_queue(self, func, q_in, q_out, window, *args, **kwargs):
        """
        Retrieve processes from the queue.

        Exceptions are captured and returned through the output queue rather
        than propagating. A worker that died without posting a result would
        leave the parent blocking on `q_out.get()` indefinitely.
        """
        while True:
            pos, xc, yc, seed = q_in.get()
            if pos is None:
                break

            pass_args = [window, xc, yc]
            pass_args.extend(args)

            call_kwargs = kwargs
            if seed is not None:
                call_kwargs = dict(kwargs, seed=seed)

            try:
                res = func(*pass_args, **call_kwargs)
            except Exception as e:
                res = _CentroidError(xc, yc, e)

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
                additional keyword arguments to pass to `func`.
                Two keys are reserved and consumed here rather than
                forwarded:

                - `on_error` : {"raise", "ignore"} (default="raise")
                  what to do when `func` fails for a centroid. `"ignore"`
                  fills that centroid with NaN and warns.
                - `seed` : int or None (default=None)
                  seeds any stochastic routine reproducibly. Each centroid
                  receives an independent child seed derived from
                  `numpy.random.SeedSequence`, passed to `func` as a `seed`
                  keyword, so results do not depend on how many processors
                  were used. `func` must accept a `seed` keyword if this is
                  supplied.

                `spectrum` is rejected outright: one spectrum describes one
                window at one centroid, so it cannot mean anything for a list
                of them.

        Returns:
            out : list of lists
                (depends on output of `func` - see notes)

        Usage:
            An obvious use case is to compute the Curie depth for many
            centroids in parallel.

            >>> self.parallelise_routine(window, xc_list, yc_list, self.optimise)

            Each centroid is assigned a new process and sent to a free processor
            to compute. In this case, the output is separate lists of shape(l,)
            for :math:`\\beta, z_t, \\Delta z, C`. If `len(xc_list)=2` then,

            >>> self.parallelise_routine(window, [x1,x2], [y1, y2], self.optimise)
            [[beta1  beta2], [zt1  zt2], [dz1  dz2], [C1  C2]]

            Another example is to parallelise the sensitivity analysis:

            >>> self.parallelise_routine(window, xc_list, yc_list, self.sensitivity, nsim)

            This time the output will be a list of lists for :math:`\\beta, z_t, \\Delta z, C`
            i.e. if `len(xc_list)=2` is the number of centroids and `nsim=4` is the number of
            simulations then separate lists will be returned for :math:`\\beta, z_t, \\Delta z, C`.

            >>> self.parallelise_routine(window, [x1,x2], [y1,y2], self.sensitivity, 4)

            which would return:

            .. code-block:: python

                [[[ beta1a , beta1b , beta1c , beta1d ],   # centroid 1 (x1,y1)
                  [ beta2a , beta2b , beta2c , beta2d ]],  # centroid 2 (x2,y2)
                 [[   zt1a ,   zt1b ,   zt1c ,   zt1d ],   # centroid 1 (x1,y1)
                  [   zt2a ,   zt2b ,   zt2c ,   zt2d ]],  # centroid 2 (x2,y2)
                 [[   dz1a ,   dz1b ,   dz1c ,   dz1d ],   # centroid 1 (x1,y1)
                  [   dz2a ,   dz2b ,   dz2c ,   dz2d ]]   # centroid 2 (x2,y2)
                 [[    C1a ,    C1b ,    C1c ,    C1d ],   # centroid 1 (x1,y1)
                  [    C2a ,    C2b ,    C2c ,    C2d ]]]  # centroid 2 (x2,y2)

        Notes:
            See the module docstring for the `if __name__ == "__main__":`
            requirement when calling this from a script.
        """

        on_error = kwargs.pop("on_error", "raise")
        if on_error not in ("raise", "ignore"):
            raise ValueError("on_error must be 'raise' or 'ignore'")

        # forwarding this would hand every centroid the same spectrum, which
        # returns the same answer at each of them -- a flat map that looks like
        # a result. The per-centroid provenance warning only fires on the
        # serial path, so whether the user is told would depend on nprocs.
        if "spectrum" in kwargs:
            raise ValueError(
                "spectrum describes a single window at a single centroid, so "
                "it cannot be shared across a list of them. Drop it and let "
                "each centroid compute its own."
            )

        seed = kwargs.pop("seed", None)

        if seed is not None and not getattr(func, "wants_seed", False):
            warnings.warn(
                "{} is deterministic, so seed has no effect on it. Only the "
                "stochastic routines -- sensitivity, metropolis_hastings -- "
                "consume one.".format(getattr(func, "__name__", func)),
                RuntimeWarning,
                stacklevel=2,
            )
            seed = None

        n = len(xc_list)
        if n != len(yc_list):
            raise ValueError("xc_list and yc_list must be the same size")
        if n == 0:
            raise ValueError("xc_list is empty, there are no centroids to evaluate")

        if seed is None:
            child_seeds = [None] * n
        else:
            # independent streams per centroid, so the result does not depend
            # on how the work happened to be distributed across processors
            child_seeds = [
                int(s.generate_state(1)[0])
                for s in np.random.SeedSequence(seed).spawn(n)
            ]

        xOpt = [None for i in range(n)]

        nprocs = self.max_processors
        if nprocs < 1:
            raise ValueError(
                "{} processors is invalid, specify a positive integer value".format(
                    nprocs
                )
            )

        unpicklable = _find_unpicklable(func, args, kwargs)
        if nprocs > 1 and unpicklable is not None:
            warnings.warn(
                "cannot send {} to worker processes under the '{}' start "
                "method. Falling back to serial evaluation. Define it at module "
                "level in an importable module to run in parallel.".format(
                    unpicklable, get_start_method()
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            nprocs = 1

        if nprocs == 1:
            # skip all the OpenMP cruft
            for i in range(n):
                xc = xc_list[i]
                yc = yc_list[i]

                call_kwargs = kwargs
                if child_seeds[i] is not None:
                    call_kwargs = dict(kwargs, seed=child_seeds[i])

                try:
                    xOpt[i] = func(window, xc, yc, *args, **call_kwargs)
                except Exception as e:
                    xOpt[i] = _CentroidError(xc, yc, e)

        else:
            # more than one processor
            processes = []
            # unbounded, so feeding the queue can never block the parent if the
            # workers die before draining it
            q_in = Queue()
            q_out = Queue()

            for i in range(nprocs):
                pass_args = [func, q_in, q_out, window]
                pass_args.extend(args)

                p = Process(target=self._func_queue, args=tuple(pass_args), kwargs=kwargs)

                processes.append(p)

            for p in processes:
                p.daemon = True
                p.start()

            # put items in the queue
            for i in range(n):
                q_in.put((i, xc_list[i], yc_list[i], child_seeds[i]))
            [q_in.put((None, None, None, None)) for _ in range(nprocs)]

            # Collect results, watching for workers that died without posting
            # one. A worker can fail before reaching our try/except -- an
            # argument that unpickles only in the parent, or the process being
            # killed -- and waiting unconditionally would deadlock.
            received = 0
            while received < n:
                try:
                    index, res = q_out.get(timeout=1.0)
                except _queue.Empty:
                    if not any(p.is_alive() for p in processes):
                        for p in processes:
                            p.join()
                        raise RuntimeError(
                            "all {} worker processes exited after returning {} of "
                            "{} results. This usually means an argument could not "
                            "be reconstructed in the worker under the '{}' start "
                            "method -- a function defined in a notebook cell or in "
                            "__main__, for instance. Define it in an importable "
                            "module, or set max_processors=1 to run "
                            "serially.".format(
                                nprocs, received, n, get_start_method()
                            )
                        )
                    continue
                xOpt[index] = res
                received += 1

            # wait until each processor has finished
            [p.join() for p in processes]

        return _collect(xOpt, n, on_error)


class _CentroidError(object):
    """A failure at one centroid, ferried back from a worker process."""

    def __init__(self, xc, yc, exception):
        self.xc = xc
        self.yc = yc
        self.exception = exception

    def __str__(self):
        return "centroid ({}, {}): {}: {}".format(
            self.xc, self.yc, type(self.exception).__name__, self.exception
        )


def _find_unpicklable(func, args, kwargs):
    """
    Return a description of the first argument a worker could not reconstruct,
    or None.

    Checked up front because such an argument kills every worker on startup,
    and the parent would otherwise wait on results that never arrive.

    Two distinct failure modes are caught. Lambdas, closures and locally
    defined functions fail `pickle.dumps` outright. Functions defined at the
    top level of `__main__` -- a script, or a notebook cell -- are subtler:
    they pickle by qualified name and so succeed *here*, but under the spawn
    start method the child has no `__main__` to look them up in, and dies on
    unpickling.
    """
    spawning = get_start_method() != "fork"

    candidates = [("func", func)]
    candidates += [("positional argument {}".format(i), a) for i, a in enumerate(args)]
    candidates += [("keyword argument '{}'".format(k), v) for k, v in kwargs.items()]

    for label, obj in candidates:
        if obj is None:
            continue

        name = getattr(obj, "__qualname__", None) or repr(obj)

        try:
            pickle.dumps(obj)
        except Exception:
            return "{} ({})".format(label, name)

        # a bound method carries its instance, whose class must also be
        # importable in the child -- check the underlying function's module
        owner = getattr(obj, "__self__", None)
        module = getattr(obj, "__module__", None)
        if owner is not None:
            module = getattr(type(owner), "__module__", module)

        if spawning and callable(obj) and module in ("__main__", "__mp_main__"):
            return "{} ({}, defined in {})".format(label, name, module)

    return None


def _collect(xOpt, n, on_error):
    """
    Reassemble per-centroid results, handling any that failed.

    The output shape is taken from the first *successful* result rather than
    whichever one happened to arrive last.
    """
    failures = [r for r in xOpt if isinstance(r, _CentroidError)]

    if failures and on_error == "raise":
        raise RuntimeError(
            "{} of {} centroids failed. First failure -- {}. Pass "
            "on_error='ignore' to fill failures with NaN instead.".format(
                len(failures), n, failures[0]
            )
        ) from failures[0].exception

    template = next((r for r in xOpt if not isinstance(r, _CentroidError)), None)
    if template is None:
        raise RuntimeError(
            "all {} centroids failed. First failure -- {}".format(n, failures[0])
        ) from failures[0].exception

    if failures:
        warnings.warn(
            "{} of {} centroids failed and were filled with NaN. "
            "First failure -- {}".format(len(failures), n, failures[0]),
            RuntimeWarning,
            stacklevel=3,
        )
        nan_fill = np.full(np.shape(template), np.nan, dtype=float)
        xOpt = [nan_fill if isinstance(r, _CentroidError) else r for r in xOpt]

    ndim = np.ndim(template)

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
