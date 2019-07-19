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
The `pycurious.mapping` module of PyCurious contains various functions to help
manipulate geospatial data into common formats. It handles commonly encounted
operations, such as:

- Gridding scattered data points
- Converting between coordinate reference systems (CRS)
- Importing and exporting GeoTiff files

It requires some **additional dependencies**:

- [`matplotlib`](https://matplotlib.org/) - for plotting
- [`pyproj`](https://github.com/jswhit/pyproj) - for transforming between different CRS
- [`cartopy`](https://scitools.org.uk/cartopy/docs/latest/) - for generating maps

Beware that most global data are georeferenced in WGS84 (EPSG: 4326).
The radial power spectrum must be in rad/km, which requires a transformation
from longitude / latitude to a local projection in eastings / northings.

For example, EMAG2 is a global compilation of the magnetic anomaly
georeferenced in WGS84 longitude / latitude. This will need to be projected
in a local CRS to use with PyCurious. If, for example, we are interested in a
region across Ireland we could use the IRENET95 local CRS (EPSG: 2157),

```python
transform_coordinates(lons, lats, epsg_in=4326, epsg_out=2157)
```

which would return a list of eastings and northings in IRENET95 projection.
"""

# -*- coding: utf-8 -*-
import numpy as np

try:
    range = xrange
except:
    pass


def transform_coordinates(x, y, epsg_in, epsg_out):
    """
    Transform between any coordinate system.

    **Requires `pyproj`** - install using pip.

    Args:
        x : float / 1D array
            x coordinates (may be in degrees or metres/eastings)
        y : float / 1D array
            y coordinates (may be in degrees or metres/northings)
        epsg_in : int
            CRS of x and y coordinates
        epsg_out : int
            CRS of output

    Returns:
        x_out : float / list of floats
            x coordinates projected in `epsg_out`
        y_out : float / list of floats
            y coordinates projected in `epsg_out`
    """
    import pyproj

    proj_in = pyproj.Proj("+init=EPSG:" + str(epsg_in))
    proj_out = pyproj.Proj("+init=EPSG:" + str(epsg_out))
    return pyproj.transform(proj_in, proj_out, x, y)


def convert_extent(extent_in, epsg_in, epsg_out):
    """
    Transform extent from epsg_in to epsg_out

    Args:
        extent_in : tuple
            bounding box [minX, maxX, minY, maxY]
        epsg_in : int
            CRS of extent
        epsg_out : int
            CRS of output

    Returns:
        extent_out : tuple
            bounding box in new CRS
    """
    xmin, xmax, ymin, ymax = extent_in
    xi = [xmin, xmin, xmax, xmax]
    yi = [ymin, ymax, ymin, ymax]
    xo, yo = transform_coordinates(xi, yi, epsg_in, epsg_out)
    extent_out = [min(xo), max(xo), min(yo), max(yo)]
    return extent_out


def trim(coords, data, extent, buffer_amount=0.0):
    """
    Trim a smaller section of a large dataset taking into
    consideration transformations into various coordinate
    reference systems (CRS).
    
    Args:
        coords : array shape (n,2)
            geographical / projected coordinates
        data : array shape (n,)
            values corresponding to coordinates
        extent : tuple
            bounding box to trim data
        buffer : float
            amount of buffer to include (default=0.0)

    Returns:
        coords_trim : array shape (l,2)
            trimmed coordinates
        data_trim : array shape (l,2)
            trimmed data array
    """
    xmin, xmax, ymin, ymax = extent

    # Extract only the data within the extent
    data_mask = np.ones(data.shape[0], dtype=bool)

    # Add a 1 percent buffer zone
    x_buffer = buffer_amount * (xmax - xmin)
    y_buffer = buffer_amount * (ymax - ymin)

    mask_e = coords[:, 0] < xmin - x_buffer
    mask_w = coords[:, 0] > xmax + x_buffer
    mask_n = coords[:, 1] < ymin - y_buffer
    mask_s = coords[:, 1] > ymax + y_buffer
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
    reference systems (CRS).

    **Requires `scipy.interpolate.griddata`**
    
    Args:
        coords : array shape (n,2)
            geographical coordinates
        data : array shape (n,) 
            values corresponding to coordinates
        extent : tuple
           bounding box in espg_out coordinates
        shape : tuple (nrows,ncols)
           size of the box, if None, shape is estimated from coords spacing
        epsg_in : int
           CRS of data (if transformation is required)
        epsg_out : int
           CRS of grid (if transformation is required)
        kwargs : keyword arguments
           keyword arguments to pass to griddata from
           `scipy.interpolate.griddata`
    
    Returns:
        grid : array shape (nrows, ncols)
            rectangular section of data bounded by extent
    """
    from scipy.interpolate import griddata

    xmin, xmax, ymin, ymax = extent

    if epsg_in is not None:
        xt, yt = transform_coordinates(
            np.array([xmin, xmin, xmax, xmax]),
            np.array([ymin, ymax, ymin, ymax]),
            epsg_out,
            epsg_in,
        )
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

    if epsg_in is not None:
        # convert back to output CRS
        xtrim, ytrim = transform_coordinates(
            coords_trim[:, 0], coords_trim[:, 1], epsg_in, epsg_out
        )
        coords_trim = np.column_stack([xtrim, ytrim])

    if shape == None:
        # estimate based on the data spacing
        xunique = np.unique(coords_trim[:, 0])
        yunique = np.unique(coords_trim[:, 1])
        dx = np.diff(xunique).mean()
        dy = np.diff(yunique).mean()
        nc = int((xtmax - xtmin) / dx)
        nr = int((ytmax - ytmin) / dy)
        print(
            "using nrows={}, ncols={} with cell spacing of {}".format(nr, nc, (dy, dx))
        )
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
    information of the Coordinate Reference System (CRS).

    **Requires `osgeo`.**

    Args:
        file_path : str
            path to the GeoTIFF

    Returns:
        data : 2D array
        extent : tuple
            bounding box in the projection of the GeoTIFF
            e.g. [xmin, xmax, ymin, ymax]
    """
    from osgeo import gdal, osr

    gtiff = gdal.Open(file_path)
    data = gtiff.ReadAsArray()
    gt = gtiff.GetGeoTransform()
    gtproj = gtiff.GetProjection()

    inproj = osr.SpatialReference()
    inproj.ImportFromWkt(gtproj)

    gtextent = (
        gt[0],
        gt[0] + gtiff.RasterXSize * gt[1],
        gt[3],
        gt[3] + gtiff.RasterYSize * gt[5],
    )

    # print projection information
    print(inproj)

    # this closes the geotiff
    gtiff = None

    return data, gtextent


def export_geotiff(file_path, array, extent, epsg):
    """
    Export a GeoTIFF from a numpy array projected in a
    predefined Coordinate Reference System (CRS).

    **Requires `osgeo`.**

    Args:
        file_path : str
            path to write the GeoTIFF
        array: 2D array
            array to save to GeoTiff
        extent : tuple
            bounding box in the projection of the GeoTIFF
            e.g. [xmin, xmax, ymin, ymax]
        epsg : int
            CRS of the GeoTIFF
            e.g. 4326 for WGS84

    """
    from osgeo import gdal, osr

    # import ogr, gdal, osr, os

    cols = array.shape[1]
    rows = array.shape[0]

    xmin, xmax, ymin, ymax = extent
    spacingX = (xmax - xmin) / cols
    spacingY = (ymax - ymin) / rows

    driver = gdal.GetDriverByName("GTiff")
    outRaster = driver.Create(file_path, cols, rows, 1, gdal.GDT_Float64)
    outRaster.SetGeoTransform((xmin, spacingX, 0, ymin, 0, spacingY))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(epsg)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    return
