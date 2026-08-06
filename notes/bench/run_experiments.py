"""
Fit every variant against the same windows, synthetic and real.

``--source synth`` fits generated windows whose answer is known, which is the
only place accuracy can be measured. ``--source wdmam`` fits the WDMAM L2 mesh
the published archive was built from, where nothing is known and the questions
are what moves, by how much, and whether the nested windows agree with each
other any better.

One process per case; the variants for a case share nothing, so the comparison
is not contaminated by any caching between them.
"""

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spectral_variants as sv  # noqa: E402

DATA = HERE / "wdmam_L2"
NEFF = DATA / "neff"

#: Each variant is one answer to "what spectrum, weighted how".
VARIANTS = {
    # the library out of the box
    "default": dict(kind="binned", taper="hanning", sigma="empirical"),
    # what the published WDMAM archive actually fitted
    "archive": dict(kind="binned", taper="hanning", sigma="empirical",
                    sigma_weight_km=8.0),
    # lean on the long wavelengths: tilt 0.5 exactly cancels the sqrt(k) the
    # bin counts put in, tilt 1.0 goes past it
    "tilt0.5": dict(kind="binned", taper="hanning", sigma="empirical", tilt=0.5),
    "tilt1.0": dict(kind="binned", taper="hanning", sigma="empirical", tilt=1.0),
    # no binning at all
    "unbinned": dict(kind="unbinned", taper="hanning"),
    # binned, but weighted by the dof actually measured rather than estimated
    # from the same data -- the control that separates taper from weighting
    "hann-dof": dict(kind="binned", taper="hanning", sigma="theory",
                     neff="hanning"),
    # multitaper, weighted by its measured dof
    "mt-NW2K3": dict(kind="binned", taper="dpss", NW=2.0, K=3, sigma="theory",
                     neff="dpss_NW2_K3"),
    "mt-NW3K5": dict(kind="binned", taper="dpss", NW=3.0, K=5, sigma="theory",
                     neff="dpss_NW3_K5"),
    # the same multitaper with the library's own sigma machinery, i.e. what a
    # user gets by swapping the taper and changing nothing else
    "mt-naive": dict(kind="binned", taper="dpss", NW=3.0, K=5, sigma="empirical"),
    # multitaper *and* the archive's high-k discount: the two do different
    # jobs -- one lowers the variance of the long wavelengths, the other
    # protects against a model error at the short ones -- so they should
    # compose rather than compete
    "mt+sw": dict(kind="binned", taper="dpss", NW=3.0, K=5, sigma="theory",
                  neff="dpss_NW3_K5", sigma_weight_km=8.0),
    # the hard cut the smooth discount replaced, for reference
    "kmax0.10": dict(kind="binned", taper="hanning", sigma="empirical", kmax=0.10),
}

FIELDS = ("beta", "zt", "dz", "C", "sigma_beta", "sigma_zt", "sigma_dz",
          "sigma_C", "chi2_red", "nres", "seconds", "onbound")

_NEFF_CACHE = {}


def neff_table(n, name):
    key = (n, name)
    if key not in _NEFF_CACHE:
        _NEFF_CACHE[key] = np.load(NEFF / f"n{n}_{name}.npy")[2]
    return _NEFF_CACHE[key]


def run_variants(sub, kmax=0.25, zt_pin=1.0):
    """Every variant against one window. Returns {name: dict of FIELDS}."""
    n = sub.shape[0]
    g = sv.fitter(zt_pin=zt_pin)
    out = {}
    for name, spec in VARIANTS.items():
        spec = dict(spec)
        kind = spec.pop("kind")
        tab = spec.pop("neff", None)
        kcut = spec.pop("kmax", kmax)
        t0 = time.perf_counter()
        try:
            if kind == "binned":
                spectrum = sv.binned(sub, kmax=kcut,
                                     neff=None if tab is None else neff_table(n, tab),
                                     **spec)
            else:
                spectrum = sv.unbinned(sub, kmax=kcut, **spec)
            x, s, extra = sv.fit(spectrum, g)
            row = dict(zip(FIELDS[:8], list(x) + list(s)))
            row["chi2_red"] = extra["chi2_red"]
            row["nres"] = extra["nres"]
            row["onbound"] = float(any("bound" in w for w in extra["warnings"]))
        except Exception as exc:  # a variant that fails should not lose the case
            row = {f: np.nan for f in FIELDS}
            row["onbound"] = -1.0
            print(f"   {name} failed: {exc}", file=sys.stderr)
        row["seconds"] = time.perf_counter() - t0
        out[name] = row
    return out


# --------------------------------------------------------------------------

def degrade(data, sigma_r_km, dx_km=5.0):
    """Blur the field to a coarser effective resolution than its grid.

    WDMAM v2.2 is an interpolated compilation: its effective resolution is far
    coarser than its 3-arcmin grid, so the observed spectrum rolls off well
    below the grid Nyquist. ``curie_config`` models that as a Gaussian, which
    multiplies the *power* spectrum by ``exp(-k**2 sigma_r**2)`` -- an
    unmodelled term the Bouligand fit has no parameter for, and which
    ``sigma_weight_km`` and ``kmax`` exist to defend against.

    Fitting clean synthetics measures a variant in a world where that defect
    does not exist, which is not the world WDMAM lives in.
    """
    n = data.shape[0]
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx_km)
    kh = np.hypot(*np.meshgrid(kx, kx, indexing="ij"))
    # amplitude transfer exp(-k^2 sigma_r^2 / 2) -> power exp(-k^2 sigma_r^2)
    return np.real(np.fft.ifft2(np.fft.fft2(data)
                                * np.exp(-0.5 * (kh * sigma_r_km) ** 2)))


NESTED = (201, 401, 801)


def _synth_case(job):
    """One field, fitted at every window -- nested, as the WDMAM mesh is.

    Independent fields per window would make the window-to-window spread a
    comparison of different realisations. Nesting them makes it the same
    quantity the WDMAM ladder reports, so `consistency` means the same thing
    on both sides and can be checked against an accuracy that is only knowable
    here.
    """
    dz_true, seed, sigma_r = job
    field = sv.synth(max(NESTED) + 10, seed, beta=3.0, zt=1.0, dz=dz_true, C=5.0)
    if sigma_r:
        field = degrade(field, sigma_r)
    out = {}
    for n in NESTED:
        for name, row in run_variants(sv.detrend(sv.centred(field, n))).items():
            out[(name, n)] = row
    return job, out


def _wdmam_case(job):
    j, w_km = job
    win = np.load(DATA / "windows_L2.npy", mmap_mode="r")
    sub = sv.detrend(sv.centred(win[j], int(w_km * 1e3 / 5000.0)))
    return job, run_variants(sub)


def collect(cases, fn, workers, label):
    t0 = time.perf_counter()
    ctx = mp.get_context("fork")
    rows = []
    with ctx.Pool(workers) as pool:
        for done, (job, res) in enumerate(pool.imap_unordered(fn, cases, chunksize=1), 1):
            for name, row in res.items():
                rows.append((job, name, row))
            if done % 25 == 0 or done == len(cases):
                el = time.perf_counter() - t0
                print(f"  {label} {done}/{len(cases)}  {el:.0f} s "
                      f"(eta {el / done * (len(cases) - done):.0f} s)", flush=True)
    return rows


def to_arrays(rows, keys):
    """Flatten to a dict of columns, one row per (case, variant).

    A variant key may be a bare name or ``(name, n)``; the second form adds an
    ``n`` column, which is how the nested synthetic windows arrive.
    """
    out = {k: [] for k in keys}
    out["variant"] = []
    nested = any(isinstance(name, tuple) for _, name, _ in rows)
    if nested:
        out["n"] = []
    for f in FIELDS:
        out[f] = []
    for job, name, row in rows:
        for k, v in zip(keys, job):
            out[k].append(v)
        if nested:
            name, n = name
            out["n"].append(n)
        out["variant"].append(name)
        for f in FIELDS:
            out[f].append(row[f])
    return {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synth", "wdmam"], required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--windows", type=int, nargs="+", default=[1000, 2000, 4000])
    ap.add_argument("--sigma-r", type=float, nargs="+", default=[0.0, 4.2],
                    help="resolution degradations to fit, km; 0 is a clean field")
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    cells = {1000: 201, 2000: 401, 4000: 801}
    if args.source == "synth":
        cases = [(dz, s, sr) for dz in (10.0, 20.0, 30.0)
                 for s in range(args.seeds) for sr in args.sigma_r]
        rows = collect(cases, _synth_case, args.workers, "synth")
        cols = to_arrays(rows, ("dz_true", "seed", "sigma_r"))
    else:
        npoints = np.load(DATA / "mesh_L2.npz")["lon"].size
        cases = [(j, w) for w in args.windows for j in range(npoints)]
        rows = collect(cases, _wdmam_case, args.workers, "wdmam")
        cols = to_arrays(rows, ("vertex", "window_km"))

    path = DATA / f"results_{args.source}.npz"
    np.savez(path, **cols)
    print(f"wrote {path}  ({len(rows)} fits)")


if __name__ == "__main__":
    main()
