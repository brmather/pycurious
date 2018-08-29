"""
Copyright 2018 Ben Mather, Robert Delhaye

This file is part of PyCurious.

PyCurious is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

PyCurious is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with PyCurious.  If not, see <http://www.gnu.org/licenses/>.
"""

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


def convert_extent(extent_in, epsg_in, epsg_out):
    """
    Transform extent from epsg_in to epsg_out

    Parameters
    ----------
     extent_in  : bounding box [minX, maxX, minY, maxY]
     epsg_in    : CRS of extent
     epsg_out   : CRS of output

    Returns
    -------
     extent_out : bounding box in new CRS
    """
    xmin, xmax, ymin, ymax = extent_in
    xi = [xmin, xmin, xmax, xmax]
    yi = [ymin, ymax, ymin, ymax]
    xo, yo = transform_coordinates(xi, yi, epsg_in, epsg_out)
    extent_out = [min(xo), max(xo), min(yo), max(yo)]
    return extent_out


def trim(coords, data, extent, buffer_amount=0.0):
    """
    Grid a smaller section of a large dataset taking into
    consideration transformations into various coordinate
    reference systems (CRS)
    
    Parameters
    ----------
     coords      : geographical / projected coordinates
     data        : values corresponding to coordinates
     extent      : box contained within the data
     buffer      : amount of buffer to include (default=0.0)

    Returns
    -------
     coords_trim : trimmed coordinates
     data_trim   : trimmed data array
    """
    xmin, xmax, ymin, ymax = extent

    # Extract only the data within the extent
    data_mask = np.ones(data.shape[0], dtype=bool)

    # Add a 1 percent buffer zone
    x_buffer = buffer_amount*(xmax - xmin)
    y_buffer = buffer_amount*(ymax - ymin)

    mask_e = coords[:,0] < xmin - x_buffer
    mask_w = coords[:,0] > xmax + x_buffer
    mask_n = coords[:,1] < ymin - y_buffer
    mask_s = coords[:,1] > ymax + y_buffer
    data_mask[mask_n] = False
    data_mask[mask_s] = False
    data_mask[mask_e] = False
    data_mask[mask_w] = False
    
    data_trim = data[data_mask]
    coords_trim = coords[data_mask]

    return coords_trim, data_trim


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

    xtextent = [xtmin, xtmax, ytmin, ytmax]

    # trim data - buffer = 5%
    coords_trim, data_trim = trim(coords, data, xtextent, 0.05)


    if type(epsg_in) != type(None):
        # convert back to output CRS
        xtrim, ytrim = transform_coordinates(coords_trim[:,0],\
                                             coords_trim[:,1],\
                                             epsg_in, epsg_out)
        coords_trim = np.column_stack([xtrim, ytrim])


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

    # interpolate

    xcoords = np.linspace(xmin, xmax, nc)
    ycoords = np.linspace(ymin, ymax, nr)
    xq, yq = np.meshgrid(xcoords, ycoords)

    vq = griddata(coords_trim, data_trim, (xq, yq), **kwargs)
    return vq


def import_geotiff(file_path):
    """
    Import a GeoTIFF to a numpy array and prints
    information of the Coordinate Reference System (CRS)

    Parameters
    ----------
     file_path : path to the GeoTIFF

    Returns
    -------
     data   : 2D numpy array
     extent : extent in the projection of the GeoTIFF
        e.g. [xmin, xmax, ymin, ymax]
    """
    from osgeo import gdal, osr
    
    gtiff = gdal.Open(file_path)
    data = gtiff.ReadAsArray()
    gt = gtiff.GetGeoTransform()
    gtproj = gtiff.GetProjection()

    inproj = osr.SpatialReference()
    inproj.ImportFromWkt(gtproj)

    gtextent = (gt[0], gt[0] + gtiff.RasterXSize*gt[1],\
                gt[3], gt[3] + gtiff.RasterYSize*gt[5])

    # print projection information
    print(inproj)

    # this closes the geotiff
    gtiff = None

    return data, gtextent


def export_geotiff(file_path, array, extent, epsg):
    """
    Export a GeoTIFF from a numpy array projected in a
    predefined Coordinate Reference System (CRS)

    Parameters
    ----------
     file_path : str, path to write the GeoTIFF
     array     : 2D numpy array
     extent    : extent in the projection of the GeoTIFF
        e.g. [xmin, xmax, ymin, ymax]
     epsg      : int, CRS of the GeoTIFF

    """
    from osgeo import gdal, osr
    # import ogr, gdal, osr, os

    cols = array.shape[1]
    rows = array.shape[0]

    xmin, xmax, ymin, ymax = extent
    spacingX = (xmax - xmin)/cols
    spacingY = (ymax - ymin)/rows

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(file_path, cols, rows, 1, gdal.GDT_Float64)
    outRaster.SetGeoTransform((xmin, spacingX, 0, ymin, 0, spacingY))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(epsg)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    return