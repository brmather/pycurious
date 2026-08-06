"""Load the cached WDMAM spectra and refit them under alternative models.

The expensive part of Global_CPD (projection, resample, FFT, radial binning) is
already done and archived in data/spectra_wdmam_L5.zarr, so any reformulation
that only changes the forward model or the estimator can be tested for the cost
of an optimiser call per vertex.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import numpy as np
import zarr
from zarr.storage import LocalStore

GCPD = "/home/unimelb.edu.au/matherb/git/../Global_CPD"
GCPD = os.path.expanduser("~/Global_CPD")
sys.path.insert(0, GCPD)
import curie_config as cfg  # noqa: E402

WINDOWS = [10000, 8000, 6000, 4000, 2000, 1000]


def archive():
    return zarr.open_group(
        store=LocalStore(f"{GCPD}/data/curie_wdmam_L5.zarr"), mode="r")


def spectra_store():
    return zarr.open_group(
        store=LocalStore(f"{GCPD}/data/spectra_wdmam_L5.zarr"), mode="r")


def load_window(w, res=None, sp=None):
    """(k, Phi, sigma, beta_prior, valid) for every vertex at window w km.

    Columns are per-vertex; rows are spectral bins, NaN-padded to a common
    length. `valid` is the mask of vertices with a usable fit in the archive.
    """
    res = res if res is not None else archive()
    sp = sp if sp is not None else spectra_store()
    g = sp[f"w{int(w)}"]
    k = np.asarray(g["k"][:], dtype=np.float64)
    Phi = np.asarray(g["Phi"][:], dtype=np.float64)
    sig = np.asarray(g["sigma"][:], dtype=np.float64)
    iw = WINDOWS.index(int(w))
    beta_prior = np.asarray(res["qc/beta_prior"][:])[iw]
    status = np.asarray(res["qc/status"][:])[iw]
    dz = cfg.read_param(res, "dz")[iw]
    FATAL = 16 | 32 | 64 | 128 | 256
    valid = (status >= 0) & ((status & FATAL) == 0) & np.isfinite(dz) \
        & np.isfinite(beta_prior) & np.isfinite(k).any(axis=0)
    return k, Phi, sig, beta_prior, valid


def archive_params(w, res=None):
    res = res if res is not None else archive()
    iw = WINDOWS.index(int(w))
    out = {}
    for name in ("beta", "zt", "dz"):
        out[name] = cfg.read_param(res, name)[iw]
    out["cpd"] = np.asarray(res["params/cpd"][:])[iw]
    out["sigma_dz"] = cfg.read_param(res, "sigma_dz")[iw]
    for name in ("land_fraction", "survey_fraction", "chi2_reduced", "beta_prior"):
        out[name] = np.asarray(res[f"qc/{name}"][:])[iw]
    return out


def mesh():
    res = archive()
    return np.asarray(res["mesh/lon"][:]), np.asarray(res["mesh/lat"][:])


def column(k, Phi, sig, j):
    """One vertex's spectrum with the NaN padding stripped."""
    kk, pp, ss = k[:, j], Phi[:, j], sig[:, j]
    m = np.isfinite(kk) & np.isfinite(pp) & np.isfinite(ss) & (kk > 0)
    return kk[m], pp[m], ss[m]
