
#ifndef __cradon_h__
#define __cradon_h__

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/ndarrayobject.h"

int radon2d(const double* Tx,
            const double* Rx,
            const size_t nTx,
            const size_t nx,
            const size_t ny,
            PyObject* L);

#endif
