"""Depth to the top of the magnetic layer, from independent data.

The workflow pins zt at a global 1.0 km. But zt is not a free parameter to be
assumed -- it is the depth from the observation plane (sea level, for WDMAM) to
the top of the magnetic basement, and that is measured:

    zt = water_depth - elevation + sediment_thickness

Water is non-magnetic, so the water column is *literally* the exp(-2 k zt)
upward continuation the model already contains; sediments are magnetically
negligible against basement. Over the ocean this is 3-7 km, not 1.

Averaged over each window with the window's own Hanning taper, because that is
what the spectrum averages over. The average is taken in zt rather than in
exp(-2k zt): the coherent average of the attenuation is the exact quantity, and
Jensen makes it >= the attenuation at mean zt, so this is a mild underestimate
of the correction, not an overestimate. It is reported in check().
"""
import os
import numpy as np

import harness as H
import curie_config as cfg

R_EARTH = cfg.EARTH_RADIUS_KM
NSAMP = 25                       # per window axis
CACHE = "zt_windows.npz"


def _laea_inverse(x_km, y_km, lon0_deg, lat0_deg):
    """Local Lambert azimuthal equal-area -> (lon, lat) in degrees.

    Closed form, vectorised over vertices: pyproj one vertex at a time would be
    ~10^8 calls for the full mesh.
    """
    lon0 = np.radians(lon0_deg)
    lat0 = np.radians(lat0_deg)
    rho = np.hypot(x_km, y_km)
    c = 2.0 * np.arcsin(np.clip(rho / (2.0 * R_EARTH), -1.0, 1.0))
    sc, cc = np.sin(c), np.cos(c)
    small = rho < 1e-9
    rho_safe = np.where(small, 1.0, rho)

    lat = np.arcsin(np.clip(cc * np.sin(lat0) + y_km * sc * np.cos(lat0) / rho_safe,
                            -1.0, 1.0))
    lon = lon0 + np.arctan2(
        x_km * sc,
        rho_safe * np.cos(lat0) * cc - y_km * np.sin(lat0) * sc)
    lat = np.where(small, lat0, lat)
    lon = np.where(small, lon0, lon)
    return np.degrees(lon), np.degrees(lat)


def zt_point(paths, lon, lat):
    """Point value of zt (km below sea level), from CRUST1.0."""
    lay = cfg.sample_crust1_layers(paths, lon, lat)
    if lay is None:
        raise RuntimeError("crust1.bnds not present")
    return lay["water_depth"] - lay["elevation"] + lay["sed_thickness"], lay


def build(paths, lon, lat, windows=H.WINDOWS, nsamp=NSAMP):
    """Taper-weighted mean zt over each window at each vertex -> (nw, nvert)."""
    u = (np.arange(nsamp) - (nsamp - 1) / 2.0) / (nsamp - 1)      # -0.5 .. 0.5
    w1 = np.hanning(nsamp + 2)[1:-1]
    wt2 = np.outer(w1, w1).ravel()
    wt2 = wt2 / wt2.sum()

    out = np.full((len(windows), lon.size), np.nan)
    for iw, W in enumerate(windows):
        acc = np.zeros(lon.size)
        gx, gy = np.meshgrid(u * W, u * W, indexing="ij")
        gx, gy = gx.ravel(), gy.ravel()
        for m in range(gx.size):
            slon, slat = _laea_inverse(gx[m], gy[m], lon, lat)
            z, _ = zt_point(paths, slon, slat)
            acc += wt2[m] * z
        out[iw] = acc
        print(f"  window {W:5d} km: median zt = {np.median(acc):5.2f} km, "
              f"IQR [{np.percentile(acc,25):.2f}, {np.percentile(acc,75):.2f}]")
    return out


def load(paths=None, lon=None, lat=None):
    if os.path.exists(CACHE):
        return np.load(CACHE)["zt"]
    if paths is None:
        paths = cfg.Paths(cfg.DEFAULT_WORKDIR, "wdmam")
        lon, lat = H.mesh()
    zt = build(paths, lon, lat)
    np.savez_compressed(CACHE, zt=zt)
    return zt


if __name__ == "__main__":
    paths = cfg.Paths(cfg.DEFAULT_WORKDIR, "wdmam")
    lon, lat = H.mesh()
    zt = load(paths, lon, lat)

    res = H.archive()
    lf = np.asarray(res["qc/land_fraction"][:])
    print("\nsanity, by regime (4000 km window):")
    iw = H.WINDOWS.index(4000)
    ocean = lf[iw] < cfg.OCEAN_MAX_LAND
    cont = lf[iw] > cfg.CONT_MIN_LAND
    print(f"  ocean      n={ocean.sum():5d}  zt = {np.median(zt[iw][ocean]):5.2f} km")
    print(f"  continent  n={cont.sum():5d}  zt = {np.median(zt[iw][cont]):5.2f} km")
    print(f"  the workflow assumes 1.00 km everywhere")

    zp, lay = zt_point(paths, lon, lat)
    for name, x, y, kind in cfg.TECTONIC_SITES:
        i = int(np.argmin((lon - x) ** 2 + (lat - y) ** 2))
        print(f"  {name:22s} {kind:7s} point zt = {zp[i]:5.2f}  "
              f"(water {lay['water_depth'][i]:.2f}, sed {lay['sed_thickness'][i]:.2f}, "
              f"elev {lay['elevation'][i]:+.2f})")
