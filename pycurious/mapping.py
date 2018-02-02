# -*- coding: utf-8 -*-
import numpy as np

try: range=xrange
except: pass

def transform_coordinates(x, y, epsg_in, epsg_out):
    """
    Transform between any coordinate system.

    Requires pyproj
    """
    import pyproj
    proj_in  = pyproj.Proj("+init=EPSG:"+str(epsg_in))
    proj_out = pyproj.Proj("+init=EPSG:"+str(epsg_out))
    return pyproj.transform(proj_in, proj_out, x, y)


def grid(coords, data, extent, shape=None, epsg_in=None, epsg_out=None, **kwargs):
    """
    Grid a smaller section of a large dataset taking into
    consideration transformations into various coordinate
    reference systems (CRS)
    
    Parameters
    ----------
     coords   : geographical coordinates
     data     : values corresponding to coordinates
     extent   : box contained within the data in espg_out
                coordinates
     shape    : size of the box (nrows,ncols)
              : if None, shape is estimated from coords spacing
     epsg_in  : CRS of data (if transformation is required)
     epsg_out : CRS of grid (if transformation is required)
     kwargs   : keyword arguments to pass to griddata from
              : scipy.interpolate.griddata
    
    Returns
    -------
     grid     : rectangular section of data bounded by extent
    """
    from scipy.interpolate import griddata
    xmin, xmax, ymin, ymax = extent
    
    if type(epsg_in) != type(None):
        xt, yt = transform_coordinates(np.array([xmin, xmin, xmax, xmax]),\
                                       np.array([ymin, ymax, ymin, ymax]),\
                                       epsg_out, epsg_in)
        # find the coordinates that will completely
        # engulf the extent
        xtmin, xtmax = min(xt), max(xt)
        ytmin, ytmax = min(yt), max(yt)
    else:
        xtmin, xtmax = xmin, xmax
        ytmin, ytmax = ymin, ymax

    
    # Extract only the data within the extent
    data_mask = np.ones(data.shape[0], dtype=bool)

    mask_e = coords[:,0] < xtmin
    mask_w = coords[:,0] > xtmax
    mask_n = coords[:,1] < ytmin
    mask_s = coords[:,1] > ytmax
    data_mask[mask_n] = False
    data_mask[mask_s] = False
    data_mask[mask_e] = False
    data_mask[mask_w] = False
    
    data_trim = data[data_mask]
    coords_trim = coords[data_mask]
    
    if shape == None:
        # estimate based on the data spacing
        xunique = np.unique(coords_trim[:,0])
        yunique = np.unique(coords_trim[:,1])
        dx = np.diff(xunique).mean()
        dy = np.diff(yunique).mean()
        nc = int((xtmax - xtmin)/dx)
        nr = int((ytmax - ytmin)/dy)
        print("using nrows={}, ncols={} with cell spacing of {}".format(nr,nc,(dy,dx)))
    else:
        nr, nc = shape
    
    
    if type(epsg_in) != type(None):
        # convert back to output CRS
        xtrim, ytrim = transform_coordinates(coords_trim[:,0],\
                                             coords_trim[:,1],\
                                             epsg_in, epsg_out)
        coords_trim = np.column_stack([xtrim, ytrim])

    # interpolate
    xcoords = np.linspace(xmin, xmax, nc)
    ycoords = np.linspace(ymin, ymax, nr)
    xq, yq = np.meshgrid(xcoords, ycoords)

    vq = griddata(coords_trim, data_trim, (xq, yq), **kwargs)
    return vq


def optimise_surfaces(surface1, surface2, sigma):
    """
    Optimise the misfit between surface1 and surface2

    surface1 and surface2 are normalised between 0 and 1
    and their residual is minimised, weighted by sigma

    Parameters
    ----------
     surface1  : starting surface (can be flat)
     surface2  : surface to match to
     sigma     : uncertainty of fitting coefficients

    Returns
    -------
     surface3  : optimised surface

    Notes
    -----
     The Krylov method uses a Krylov approximation for the
     inverse Jacobian as it is suitable for large problems
    """
    from scipy.optimize import root
    
    def objective_function(x, x0, sigma_x0):
        return (x - x0)**2/sigma_x0**2
    
    sigma = sigma.ravel()
    
    s1 = surface1.flatten()
    s1 -= s1.min()
    s1 /= s1.max()
    
    s2 = surface2.flatten()
    s2 -= s2.min()
    s2 /= s2.max()
    
    # starting point should be at prior
    x0 = s1
    
    sol = root(objective_function, x0, method='krylov')
    return sol.x.reshape(surface1.shape)
