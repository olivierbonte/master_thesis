# %% Load in
import rioxarray
import rasterio
import xarray as xr
import numpy as np
import hvplot.xarray
import hvplot.dask
from pathlib import Path
import os
import zipfile

pad = Path(os.getcwd())
if pad.name == "preprocessing_files":
    pad_correct = pad.parent
    os.chdir(pad_correct)
os.system('zenodo_get 10.5281/zenodo.7688784')
with zipfile.ZipFile("data_github.zip", 'r') as zip_ref:
    zip_ref.extractall('data_github')

def print_raster(raster):
    print(
        f"shape: {raster.rio.shape}\n"
        f"resolution: {raster.rio.resolution()}\n"
        f"bounds: {raster.rio.bounds()}\n"
        f"CRS: {raster.rio.crs}\n"
    )
# landuse = rioxarray.open_rasterio('data/Zwalm_bodembedekking'+ #type:ignore
# '/wetransfer_landgebruik_2022-11-07_0921/'+
# 'Landuse_Vlaanderen_Wallonie_final.sdat')
landuse = rioxarray.open_rasterio('data_github/Landuse_Vlaanderen_Wallonie_final.sdat')#type:ignore
landuse = landuse.chunk('auto')#type:ignore
landuse_nonan = landuse.where(landuse != 255)

#####################################################
# %% RESAMLING FOR SENTINEL: nearest neighbour resmpling of landuse => Sentinel
########################################################
# see algorithms possible:  https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling  

s1_full = xr.open_dataset('data/s0_OpenEO/S0_zwalm.nc', decode_coords='all')#type:ignore
s1_full= s1_full.rio.write_crs(32631, inplace = True) # type: ignore #manually set crs
s1_gamma0_full = xr.open_dataset('data/g0_OpenEO/g0_zwalm.nc')
s1_gamma0_full= s1_gamma0_full.rio.write_crs(32631, inplace = True) # type: ignore #manually set crs

print("Sentinel Raster:\n----------------\n")
print_raster(s1_full)
print("Landuse Raster:\n----------------\n")
print_raster(landuse)

landuse_reprojected = landuse_nonan.rio.reproject_match(
    s1_full, resampling=rasterio.enums.Resampling.nearest#type:ignore
)
#assign the coordinates from Sentiel-1 raster to avoid problems with floating point errors
landuse_reprojected = landuse_reprojected.assign_coords({
    "x":s1_full.x,
    "y":s1_full.y,
})
#landuse back to uint8 (more data efficient) and in Sentinel 1
landuse_reprojected = landuse_reprojected.astype(np.uint8)
s1_full['landuse'] = landuse_reprojected.isel(band = 0) #drop the band
s1_gamma0_full['landuse'] = landuse_reprojected.isel(band = 0) #drop the band

#%% Write landuse and S1 to new NetCDF
s1_full.to_netcdf('data/s0_OpenEO/S0_zwalm_landuse.nc', mode = 'w')
# s1_full.close()
# s1_gamma0_full.close()
# s1_full.to_netcdf('data/s0_OpenEO/S0_zwalm.nc', mode = 'a') #append to previous dataframe to save space
s1_gamma0_full.to_netcdf('data/g0_OpenEO/g0_zwalm_landuse.nc', mode = 'w')
#############################################
# %% RESAMPLING FOR LAI: 
############################################

LAI_xr = xr.open_dataset('data/LAI/LAI_cube_Zwalm.nc', decode_coords= 'all')#type:ignore
LAI_xr = LAI_xr.rio.write_crs(4326, inplace = True)#type:ignore

print("LAI Raster:\n----------------\n")
print_raster(LAI_xr)
print("Landuse Raster:\n----------------\n")
print_raster(landuse)

#Step 1: reproject landuse to EPSG:4326 with neirest neighbour
# keep same number of gridcells when doing so
landuse_4326 = landuse_nonan.rio.reproject(
    dst_crs = "EPSG:4326", shape = landuse_nonan.shape[1:3],
    resampling = rasterio.enums.Resampling.nearest#type:ignore
)
print("Landuse Raster reprojected:\n----------------\n")
print_raster(landuse_4326)

#Step 2: blockprocessing idea = take most frequently occuring category per block
landuse_4326_matched = landuse_4326.rio.reproject_match(
    LAI_xr, resampling=rasterio.enums.Resampling.mode#type:ignore
)
print("Landuse Raster reprojected and matched:\n----------------\n")
print_raster(landuse_4326_matched)

#assign the coordinates from LAI raster to avoid problems with floating point errors
landuse_4326_matched = landuse_4326_matched.assign_coords({
    "x":LAI_xr.x,
    "y":LAI_xr.y,
})
landuse_4326_matched = landuse_4326_matched.astype(np.uint8)
LAI_xr['landuse'] = landuse_4326_matched.isel(band = 0)

# %% Write landuse and LAI to new NetCDF file
LAI_xr.to_netcdf('data/LAI/LAI_cube_Zwalm_landuse.nc', mode = 'w')
