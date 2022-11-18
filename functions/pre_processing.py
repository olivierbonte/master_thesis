
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
import os
import pandas as pd
from shapely import geometry
from shapely.ops import voronoi_diagram


#######################
# Sentinel 1 Processing
#######################

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
add_dims = True, filepath_nc_raw = None, remove_nan = False):
    """Converts GeoTIFF to NetCDF with GDAL
    
    Parameters
    ---------
    filepath_input_tiff: str

    fileptah_output_nc: str 

    return_bool: bool, default = True
        if True, an xarray is returned!
    add_dims: bool, default = True
        if True, adds a time, orbit and satellite dimension to the NetCDF output
    filepath_nc_raw: pathlib.Path, default = None
        location of raw Sentinel data for info on date, only required
        if add_dims is True
    remove_nan: bool, default = False
        if True, NetCDF containing only nans for g0vv will be removed

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
    if add_dims:
        if filepath_nc_raw is None:
            raise TypeError("""filepath_nc_raw can not be None when add_time is
            True, supply raw filepath as pathlib.Path""")
        year = int(filepath_nc_raw.name[0:4])
        month = int(filepath_nc_raw.name[4:6])
        day = int(filepath_nc_raw.name[6:8])
        if "_A_" in filepath_nc_raw.name:
            hour = 18
        if "_D_" in filepath_nc_raw.name:
            hour = 6
        else:
            raise ValueError('No _A_ or _D_ found in filename, check filename')
        xarr_ds2 = xarr_ds2.expand_dims(time = [datetime.datetime(year, month, day, hour)])
        #xarr_ds2 = xarr_ds2.expand_dims(orbit = [np.array(filepath_nc_raw.name[-6:-3], dtype = np.int32)])
        #xarr_ds2 = xarr_ds2.expand_dims(satellite = [filepath_nc_raw.name[9:12]])
        #Do not change dimensions!! just add these as variables
        xarr_ds2['satellite'] = xr.DataArray(
            data = np.array([filepath_nc_raw.name[9:14]]), 
            dims = ('time'), 
            coords={"time":[datetime.datetime(year, month, day, hour)]}
        )
        xarr_ds2['orbit'] = xr.DataArray(
            data = [np.array(filepath_nc_raw.name[-6:-3], dtype = np.int32)], 
            dims = ('time'), 
            coords={"time":[datetime.datetime(year, month, day, hour)]}
        )
    xarr_ds2.to_netcdf(filepath_output_nc, mode = 'w')
    if return_bool:
        return xarr_ds2
    xarr_ds2.close()
    if remove_nan:
        if np.isnan(xarr_ds2['g0vv'].values).all():
            os.remove(filepath_output_nc)





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
    out_image, out_transform = rasterio.mask.mask(raster_rio_tiff, geom, invert = False, #type: ignore
    crop = True)
    if len(out_image.shape) == 2:
        count = 1 #number of bands
        height = out_image.shape[0]
        width = out_image.shape[1]
    if len(out_image.shape) == 3:
        if out_image.shape[0] == 3: 
            count = out_image.shape[0]
            flattened = out_image[0,:,:]
        #test_for_processing_landuse
        else:
            count = out_image.shape[0]
            flattened = out_image[0,:,:]
        height = flattened.shape[0]
        width = flattened.shape[1]
    else:
        raise Warning('Image has less than 2 or more than 3 dimensions, check raster input')
    
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
filepath_nc_processed, filepath_temp_data, epsg, return_bool = False, remove_nan = False):
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
    return_bool: bool, default = False
        if True, returns the processed xarray.Dataset
    remove_nan: bool, default = False
        if True, NetCDF containing only nans for g0vv will be removed

    Returns
    --------
    cf. return bool


    """
    ds = read_netcdf(filepath_nc_raw, epsg)
    filepath_out = filepath_temp_data + '/' + filepath_nc_raw.name + '_raw.tiff'
    raster_rio_tiff = netcdf_to_tiff(ds, filepath_out)
    filepath_masked_out = filepath_temp_data + '/' + filepath_nc_raw.name + '_masked.tiff'
    filepath_temp_nc = filepath_temp_data + '/' + filepath_nc_raw.name + '_masked.nc'
    mask_tiff_with_shape(raster_rio_tiff, filepath_shapefile, filepath_masked_out)
    if return_bool:
        masked_xarr = tiff_to_netcdf(filepath_masked_out, filepath_nc_processed, filepath_temp_nc,return_bool,
        add_dims= True,filepath_nc_raw = filepath_nc_raw, remove_nan=remove_nan)
        return masked_xarr
    else:
        tiff_to_netcdf(filepath_masked_out, filepath_nc_processed, filepath_temp_nc, return_bool,
        add_dims= True, filepath_nc_raw = filepath_nc_raw, remove_nan=remove_nan)

####################
# Landuse processing 
####################

def custom_blockproc(im, block_sz):
    """Custom python function to extract the most occuring value per block.

    Parameters
    ---------
    im: numpy.array
        array of the image to be blockprocessed
    block_sz: tuple
        number of blocks to consider in [x,y] 
    """
    h, w = im.shape
    m, n = block_sz
    im_new = np.empty((int(h/2),int(w/2)))
    x_iter = 0
    y_iter = 0
    for x in range(0, h, m):
        for y in range(0, w, n):
            block = im[x:x+m, y:y+n]
            values, counts = np.unique(block, return_counts=True)
            ind = np.argmax(counts)
            im_new[x_iter,y_iter] = values[ind]
            y_iter = y_iter + 1
        x_iter = x_iter + 1
        y_iter = 0
    return im_new

###################
# Thiessen polygons
##################

def custom_thiessen_polygons(gdf_info, box_shape, gdf_catchment):
    """Process outpout of shapely voronoi so that only shape of catchment is considered

    Parameters
    ----------
    gdf_info: geopandas.GeoDataFrame
        Each row is a station, 3 columns must be present:
        - name: str
        - station_name: str
        - geometry: contains location as point (shapely)
    box_shape: shapely.Polygon
        Polygon defining the shape within the thiessen polygons need to be 
        constructed 
    gdf_catchment: geopandas.GeoDataFrame
        must contain 'geometry' column with the polygon shape of the catchment

    Returns
    --------
    gdf_thiessen_catchment: geopandas.GeoDataFrame   
        geometry column containing the Thiessen polygons within the catchment, also 
        area and relative area included 
    """
    points = geometry.MultiPoint(gdf_info['geometry'])
    vonoroi_shapely = voronoi_diagram(points, box_shape) ##!!WERKT VOOR 2 PUNTEN!!
    gdf_info = gdf_info.rename(columns= {'geometry':'location'})#type:ignore
    gdf_thiessen = gdf_info.copy()
    #creating the geom_list is crucial to assign correct polygon to correct name!
    geom_list = [None] * len(gdf_info['location'])
    geomss = list(vonoroi_shapely.geoms)#type:ignore
    for i in range(len(geomss)):
        for j in range(len(gdf_info['location'])):
            if geomss[i].contains(gdf_info['location'][j]):
                geom_list[j] = geomss[i]
    gdf_thiessen['geometry'] = geom_list
    gdf_thiessen['geometry'] = gdf_thiessen['geometry'].astype('geometry')
    gdf_thiessen = gdf_thiessen.set_crs('EPSG:31370')
    gdf_thiessen_catchment = gdf_thiessen.overlay(gdf_catchment[['geometry']], how='intersection')#type:ignore
    gdf_thiessen_catchment['Area'] = gdf_thiessen_catchment.area
    gdf_thiessen_catchment['relative_area'] = gdf_thiessen_catchment['Area']/np.sum(gdf_thiessen_catchment['Area'])
    return gdf_thiessen_catchment

#######################################
# Non geographic preprocessing in pandas
########################################

def make_pd_unique_timesteps(pddf, t_column_name, t_start, t_end, freq):
    """Transform pandas Dateframe so that you left join it on a
    desired timeseries going from t_start to t_end at desired frequency

    Parameters
    ------------
    pddf: pandas.DataFrame

    t_colum_name: string
        name of where time (as datetime) is located in pd
    t_start: datetime
    t_end: datetime
    freq: string
        desired frequency of timeseries
        
    Returns
    --------
    pddf_unique: pandas.DataFrame  
        unique pandas Dataframe
    """
    timeseries = pd.DataFrame({t_column_name:pd.date_range(
    t_start, t_end, freq = freq)})#type:ignore
    pddf_unique = timeseries.merge(
        pddf, on = t_column_name, how = 'left'
    )
    return pddf_unique

