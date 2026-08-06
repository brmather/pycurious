"""
Cut the WDMAM L2 mesh windows once and cache them, so the spectral experiments
downstream never pay for the projection again.

Reproduces exactly what ``03_compute_curie.py`` does per vertex -- band-limited
x4 upsample of the source grid, LAEA frame at the vertex, ``map_coordinates``
onto the master projected axis -- but stops at the window instead of fitting
it, and keeps only the central 811 cells (4055 km), which covers every window
up to the 4000 km rung.

Writes ``windows_L2.npy`` (162, 811, 811) float32, ~427 MB.
"""

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import zarr
from zarr.storage import LocalStore

CPD = Path.home() / "Global_CPD"
sys.path.insert(0, str(CPD))
import curie_config as cfg  # noqa: E402

OUT = Path(__file__).resolve().parent / "wdmam_L2"
N_MASTER = 811          # 4055 km at 5 km cells; 4000 km window is 801
DX_M = 5000.0
UPSAMPLE = 4
WORKERS = 16

_G = {}


def _init(state):
    _G.update(state)


def _cut(j):
    lon, lat = cfg.inverse_project(_G["X"], _G["Y"], _G["lon"][j], _G["lat"][j], "laea")
    win = cfg.sample_global(_G["dense"], lon, lat, _G["dense_grid"])
    return j, win.astype(np.float32)


def main():
    OUT.mkdir(exist_ok=True)
    res = zarr.open_group(store=LocalStore(str(CPD / "data/curie_wdmam_L2.zarr")), mode="r")
    lon = np.asarray(res["mesh/lon"][:])
    lat = np.asarray(res["mesh/lat"][:])
    npoints = lon.size

    src = zarr.open_group(store=LocalStore(str(CPD / "data/wdmam.zarr")), mode="r")
    a = src.attrs
    if "grid_spec" in a:
        grid = cfg.GridSpec.from_dict(dict(a["grid_spec"])).check()
    else:
        grid = cfg.GridSpec(name="wdmam", nlat=int(a["nlat"]), nlon=int(a["nlon"]),
                            dll=float(a["dll_deg"]), registration="gridline").check()
    print(f"source {grid.name}: {grid.nlat}x{grid.nlon} at {grid.dll} deg "
          f"({grid.spacing_km:.2f} km), registration {grid.registration}")

    field = np.asarray(src["anomaly_iso"][:], dtype=np.float32)
    t0 = time.perf_counter()
    dense = cfg.band_limited_upsample(field, factor=UPSAMPLE, pixel=grid.pixel)
    err = np.abs(dense[::UPSAMPLE, ::UPSAMPLE] - field).max()
    assert err < 1e-6, f"upsample does not reproduce its input nodes ({err:.2e})"
    dense = cfg.wrap_pad(dense)
    print(f"upsample x{UPSAMPLE} -> {dense.shape} {dense.nbytes / 1e9:.2f} GB "
          f"in {time.perf_counter() - t0:.0f} s")

    ax = cfg.master_axis(N_MASTER, DX_M)
    X, Y = np.meshgrid(ax, ax)
    state = {"dense": dense, "dense_grid": grid.dense(UPSAMPLE),
             "lon": lon, "lat": lat, "X": X, "Y": Y}

    out = np.lib.format.open_memmap(OUT / "windows_L2.npy", mode="w+",
                                    dtype=np.float32, shape=(npoints, N_MASTER, N_MASTER))
    t0 = time.perf_counter()
    ctx = mp.get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init, initargs=(state,)) as pool:
        for done, (j, win) in enumerate(pool.imap_unordered(_cut, range(npoints)), 1):
            out[j] = win
            if done % 40 == 0:
                print(f"  {done}/{npoints}  {time.perf_counter() - t0:.0f} s", flush=True)
    out.flush()

    np.savez(OUT / "mesh_L2.npz", lon=lon, lat=lat,
             window_km=np.asarray(res["window_km"][:]),
             status=np.asarray(res["qc/status"][:]),
             dz=np.asarray(res["params/dz"][:]).astype(np.float64) / 100.0,
             zt=np.asarray(res["params/zt"][:]).astype(np.float64) / 100.0,
             beta=np.asarray(res["params/beta"][:]).astype(np.float64) / 100.0,
             sigma_dz=np.asarray(res["params/sigma_dz"][:]),
             cpd=np.asarray(res["params/cpd"][:]))
    print(f"wrote {OUT/'windows_L2.npy'} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
