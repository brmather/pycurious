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

    # Add a 1 percent buffer zone
    x_buffer = 0.01*(xtmax - xtmin)
    y_buffer = 0.01*(ytmax - ytmin)

    mask_e = coords[:,0] < xtmin - x_buffer
    mask_w = coords[:,0] > xtmax + x_buffer
    mask_n = coords[:,1] < ytmin - y_buffer
    mask_s = coords[:,1] > ytmax + y_buffer
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