
import rasterio
import rioxarray
import numpy as np
import xarray
from pathlib import Path
import os
pad = Path(os.getcwd())
if pad.name != "Python":
    pad_correct = Path("../../../Python")
    os.chdir(pad_correct)
from functions.pre_processing import custom_blockproc

# #uitgevoerd in Lambert72
# bodembedekking_pad = Path("data/Zwalm_bodembedekking")
# landuse = rasterio.open(bodembedekking_pad/"Landuse_for_Zwalm.tif")
# mask_tiff_with_shape(raster_rio_tiff = landuse, 
#     filepath_shapefile = Path("data\Zwalm_shape\OS266.shp"),
#     filepath_out= bodembedekking_pad/"bodembekking_masked.tif",
#     nodata = 255) #use 255 since this does not occur


#landuse_WGS_84 = xarray.open_dataarray('data/Zwalm_bodembedekking/QGIS_project/Landuse_directly_to_Sentinel_resolution.tif',
#engine = "rasterio")
landuse_WGS_84 = rioxarray.open_rasterio('data/Zwalm_bodembedekking/QGIS_project/Landuse_directly_to_Sentinel_resolution.tif')#type:ignore
landuse_WGS_84 = landuse_WGS_84['band' == 1] #drop _band
landuse_WGS_84 = landuse_WGS_84.rename({'x': 'lon','y': 'lat'})
#blockprocessing with 2 x 2 kernel
landuse_WGS_84_blocked = custom_blockproc(landuse_WGS_84.values, (2,2))
#redfine the lat and lon coordinates
landuse_lat = np.arange(min(landuse_WGS_84.lat.values)+0.05/1008, max(landuse_WGS_84.lat.values)-0.05/1008, 0.2/1008)
landuse_lon = np.arange(min(landuse_WGS_84.lon.values)+0.05/1008, max(landuse_WGS_84.lon.values)-0.05/1008, 0.2/1008)

zwalm_cube = xarray.open_dataset('data/xarray_zwalm_cube.nc')
print(all(np.isclose(zwalm_cube.lon.values, landuse_lon, atol = 1e-15)))#type:ignore
print(all(np.isclose(zwalm_cube.lat.values, landuse_lat, atol = 1e-15)))#type:ignore
print(all(zwalm_cube.lon.values == landuse_lon))
#the coordinates are approximately equal, but vary due to floating point errors
# => assign manually the coordinates of the zwalm cube!
latmesh, lonmesh = np.meshgrid(np.flip(zwalm_cube.lat.values),zwalm_cube.lon.values)
landuse_WGS_84_blocked = xarray.DataArray(
    data=landuse_WGS_84_blocked.astype(np.int32),
    dims=["lat", "lon"],
    coords={
        "lat":np.flip(zwalm_cube.lat.values.tolist()), #flip needed because origin is upper left
        "lon":zwalm_cube.lon.values.tolist()
    }
)
zwalm_cube_landuse = zwalm_cube.copy()
zwalm_cube_landuse['landuse'] = landuse_WGS_84_blocked
zwalm_cube_landuse.to_netcdf('data/xarray_zwalm_landuse_cube.nc', mode ='w')

