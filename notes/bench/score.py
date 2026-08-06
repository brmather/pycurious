"""Score a CPD map exactly the way Global_CPD scores its own.

Same regime split (OCEAN_MAX_LAND / CONT_MIN_LAND), same per-vertex heat-flow
aggregation (one median q per vertex, never CPD interpolated to sites -- that
inflates n by pseudo-replication), same Spearman, same tectonic site list.
"""
import numpy as np
import scipy.stats as st

import harness as H
import curie_config as cfg

_cache = {}


def context():
    if "ctx" in _cache:
        return _cache["ctx"]
    lon, lat = H.mesh()
    paths = cfg.Paths(cfg.DEFAULT_WORKDIR, "wdmam")
    age = cfg.sample_age(paths, lon, lat)
    if age is None:
        age = np.full(lon.size, np.nan)

    hf = cfg.load_heatflow(paths)
    qmed = np.full(lon.size, np.nan)
    if hf is not None:
        qlon, qlat, q = hf.lon, hf.lat, hf.q
        from scipy.spatial import cKDTree

        def xyz(lo, la):
            lo, la = np.radians(lo), np.radians(la)
            return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]

        tree = cKDTree(xyz(lon, lat))
        _, idx = tree.query(xyz(qlon, qlat))
        order = np.argsort(idx)
        idx_s, q_s = idx[order], np.asarray(q)[order]
        bounds = np.searchsorted(idx_s, np.arange(lon.size + 1))
        for v in range(lon.size):
            a, b = bounds[v], bounds[v + 1]
            if b > a:
                qmed[v] = np.median(q_s[a:b])

    res = H.archive()
    lf = np.asarray(res["qc/land_fraction"][:])
    _cache["ctx"] = dict(lon=lon, lat=lat, age=age, q=qmed, lf=lf)
    return _cache["ctx"]


def spear(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 12:
        return np.nan
    return float(st.spearmanr(x[m], y[m])[0])


def score(cpd, w):
    """cpd: (nvert,) for one window. Returns the workflow's own test battery."""
    ctx = context()
    iw = H.WINDOWS.index(int(w))
    lf = ctx["lf"][iw]
    ocean = np.isfinite(cpd) & (lf < cfg.OCEAN_MAX_LAND)
    cont = np.isfinite(cpd) & (lf > cfg.CONT_MIN_LAND)

    out = {
        "rho_age_ocean": spear(ctx["age"][ocean], cpd[ocean]),
        "rho_q_ocean": spear(ctx["q"][ocean], cpd[ocean]),
        "rho_q_cont": spear(ctx["q"][cont], cpd[cont]),
        "n_ocean": int(ocean.sum()),
        "n_cont": int(cont.sum()),
    }

    lon, lat = ctx["lon"], ctx["lat"]
    by_kind = {}
    for name, x, y, kind in cfg.TECTONIC_SITES:
        d = (np.cos(np.radians(lat)) * ((lon - x + 180) % 360 - 180)) ** 2 + (lat - y) ** 2
        by_kind.setdefault(kind, []).append(cpd[int(np.argmin(d))])
    for kind, vals in by_kind.items():
        out[f"site_{kind}"] = float(np.nanmedian(vals))
    if "craton" in by_kind and "ridge" in by_kind:
        out["craton_ridge"] = out["site_craton"] / out["site_ridge"]
    out["median_cpd"] = float(np.nanmedian(cpd[np.isfinite(cpd)]))
    return out


HEAD = ["rho_age_ocean", "rho_q_ocean", "rho_q_cont", "site_craton",
        "site_ridge", "craton_ridge", "median_cpd"]


def table(rows, title):
    print(f"\n{title}")
    print(f"{'window':>7} " + " ".join(f"{h:>13}" for h in HEAD))
    for w, r in rows:
        print(f"{w:7d} " + " ".join(
            f"{r.get(h, float('nan')):13.3f}" for h in HEAD))
