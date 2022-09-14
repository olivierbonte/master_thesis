from calendar import month
import xarray as xr
from pathlib import Path
import rasterio
import rioxarray
import warnings
import numpy as np
from osgeo import gdal
import geopandas as gpd
import datetime
import netCDF4

def read_netcdf(filepath, epsg, transpose = False):
    """Read in 1 netCDF file and add geographic information

    Parameters
    ----------
    filepath: pathlib.Path (#str ## (to delete) | )
        location of the netCDF file
    epsg: int
        EPSG code of the desired CRS
    transpose: bool, defualt = False
        transpose lat and lon of Data variables in xarray

    Returns
    -------
    ds: xarray.DataSet
    """
    ds = xr.open_dataset(filepath)
    if ds.rio.crs is not None:
        warnings.warn("CRS was already defined, but will be overwritten")
    ds = ds.rio.write_crs("EPSG:"+str(epsg),inplace = True)
    if transpose:
        ds = ds.transpose('lat','lon')
    return ds

def netcdf_to_tiff(xarr_ds,filepath_out,transpose = True):
    """Converts NetCDF files to GeoTIFF for GDAL processing. Includes all
    data variables of the xarray

    Parameters
    ----------
    xarr_ds: xarray.DataSet
        ideally as obtained from `functions.pre_processing.read_netcdf'
    filepath_out: str | pathlit.Path
        Location to write GeoTIFF to
    transpose: bool, default = True
        transpose lat and lon of Data variables in xarray. Set to False if 
        already applied in earlier stage

    Returns
    ------
    raster_rio_tiff: rasterio.io.DatasetReader
        GeoTIFF opened with rasterio.open()
    """
    xarr_ds = xarr_ds.rio.set_spatial_dims(x_dim = 'lon', y_dim = 'lat')
    if transpose:
        xarr_ds = xarr_ds.transpose('lat','lon')
    if xarr_ds.rio.crs is None:
        raise ValueError("""crs is not yet set. Apply .rio.write_crs() with
        correct CRS on xarr_ds before applying this function""")
    xarr_ds.rio.to_raster(filepath_out)  
    raster_rio_tiff = rasterio.open(filepath_out)
    return raster_rio_tiff

def tiff_to_netcdf(filepath_input_tiff, filepath_output_nc, filepath_temp_nc, return_bool = True,
add_time = True, filepath_nc_raw = None):
    """Converts GeoTIFF to NetCDF with GDAL
    
    Parameters
    ---------
    filepath_input_tiff: str

    fileptah_output_nc: str 

    return_bool: bool, default = True
        if True, an xarray is returned!
    add_time: bool, default = True
        if True, adds a time dimension to the NetCDF output
    filepath_nc_raw: pathlib.Path, default = None
        location of raw Sentinel data for info on date, only required
        if add_time is True

    Returns
    -------
    xarr_ds2: xarray.Dataset
    """
    #fix attempt: do translate to temporary 
    ds = gdal.Translate(filepath_temp_nc, filepath_input_tiff, format = 'NetCDF')
    ds = None
    xarr_ds = xr.open_dataset(filepath_temp_nc)#, engine = 'netcdf4')
    xarr_ds.close()
    xarr_ds2 = xarr_ds.rename({"Band1":"g0vv","Band2":"g0vh","Band3":"lia"})
    if add_time:
        if filepath_nc_raw is None:
            raise TypeError("""filepath_nc_raw can not be None when add_time is
            True, supply raw filepath as pathlib.Path""")
        year = int(filepath_nc_raw.name[0:4])
        month = int(filepath_nc_raw.name[4:6])
        day = int(filepath_nc_raw.name[6:8])
        xarr_ds2 = xarr_ds2.expand_dims(time = [datetime.datetime(year, month, day)])
    #import pdb; pdb.set_trace(),
    xarr_ds2.to_netcdf(filepath_output_nc, mode = 'w')
    if return_bool:
        return xarr_ds2
    xarr_ds2.close()



def mask_tiff_with_shape(raster_rio_tiff, filepath_shapefile, filepath_out, nodata = -9999):
    """GeoTIFF, opended with rasterio, is masked and clipped with a provided
    shapefile. Important that both have the same CRS.

    Parameters
    ----------
    raster_rio_tiff:  rasterio.io.DatasetReader
        GeoTIFF opened with rasterio.open()
    shapefile: str | pathlib.Path
        Filepath of shapefile of the considered catchment
    filepath_out: str | pathlib.Path
        Location to write the clipped and masked GeoTIFF to
    nodata: int | float, default = -9999

    Returns
    -------
    masked_tiff: rasterio.io.DatasetWrtiter        
            GeoTIFF after clipping and masking
    """
    gpd_df = gpd.read_file(filepath_shapefile)
    geom = gpd_df['geometry']
    out_image, out_transform = rasterio.mask.mask(raster_rio_tiff, geom, invert = False, 
    crop = True)
    if len(out_image.shape) == 2:
        count = 1 #number of bands
        height = out_image.shape[0]
        width = out_image.shape[1]
    if len(out_image.shape) == 3:
        if out_image.shape[0] == 3: 
            count = out_image.shape[0]
            flattened = out_image[0,:,:]
        height = flattened.shape[0]
        width = flattened.shape[1]
    
    masked_tiff = rasterio.open(
        filepath_out,
        'w',
        driver = 'GTiff',
        height = height,
        width = width,
        count = count,
        dtype = out_image.dtype,
        crs = raster_rio_tiff.crs, 
        transform = out_transform,
        nodata = nodata
    )
    masked_tiff.write(out_image)
    masked_tiff.close()
    raster_rio_tiff.close()
    return masked_tiff

def pre_processing_pipeline(filepath_nc_raw, filepath_shapefile,
filepath_nc_processed, filepath_temp_data, epsg, return_bool = False):
    """Function encapsulating entire pipeline of from raw NetCDF to clipped and
    masked NetCDF according to to given shapefile.

    Parameters
    ----------
    filepath_nc_raw:  pathlib.Path
        Location of the raw Sentinel 1 NetCDF files
    filepath_shapefile: str | pathlib.Path
        Location of shapefile to mask tje g0vv image with
    filepath_nc_processed: str 
        Location where to store the masked and clipped Sentinel 1 NetCDF files
    filepath_temp_data: str 
        Location where to store the temporary data
    epsg: int
        epsg code of the desired CRS

    Returns
    --------
    return_bool: bool, default = False
        if True, returns the processed xarray.Dataset


    """
    ds = read_netcdf(filepath_nc_raw, epsg)
    filepath_out = filepath_temp_data +'/raw.tiff'
    raster_rio_tiff = netcdf_to_tiff(ds, filepath_out)
    filepath_masked_out = filepath_temp_data + '/masked.tiff'
    filepath_temp_nc = filepath_temp_data + '/masked.nc'
    mask_tiff_with_shape(raster_rio_tiff, filepath_shapefile, filepath_masked_out)
    if return_bool:
        masked_xarr = tiff_to_netcdf(filepath_masked_out, filepath_nc_processed, filepath_temp_nc,return_bool,
        add_time= True,filepath_nc_raw = filepath_nc_raw)
        return masked_xarr
    else:
        tiff_to_netcdf(filepath_masked_out, filepath_nc_processed, filepath_temp_nc, return_bool,
        add_time= True, filepath_nc_raw = filepath_nc_raw)