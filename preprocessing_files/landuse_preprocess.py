# %% Load in
import rioxarray
import rasterio
import xarray as xr
import numpy as np
import hvplot.xarray
import hvplot.dask
from pathlib import Path
import os
pad = Path(os.getcwd())
if pad.name != "Python":
    pad_correct = Path("../../Python")
    os.chdir(pad_correct)

def print_raster(raster):
    print(
        f"shape: {raster.rio.shape}\n"
        f"resolution: {raster.rio.resolution()}\n"
        f"bounds: {raster.rio.bounds()}\n"
        f"CRS: {raster.rio.crs}\n"
    )
landuse = rioxarray.open_rasterio('data/Zwalm_bodembedekking'+ #type:ignore
'/wetransfer_landgebruik_2022-11-07_0921/'+
'Landuse_Vlaanderen_Wallonie_final.sdat')
landuse = landuse.chunk('auto')#type:ignore
landuse_nonan = landuse.where(landuse != 255)

# %% Resampling: nearest neighbour resmpling of landuse => Sentinel
# see algorithms possible:  https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling  

s1_full = rioxarray.open_rasterio('data/s0_OpenEO/S0_zwalm.nc')#type:ignore
s1_full= s1_full.rio.write_crs(32631, inplace = True) # type: ignore #manually set crs

print("Sentinel Raster:\n----------------\n")
print_raster(s1_full)
print("Landuse Raster:\n----------------\n")
print_raster(landuse)

landuse_reprojected = landuse_nonan.rio.reproject_match(
    s1_full, resampling=rasterio.enums.Resampling.nearest#type:ignore
)
#assign the coordinates from Sentiel-1 raster to avoid problems with floating point erros
landuse_reprojected = landuse_reprojected.assign_coords({
    "x":s1_full.x,
    "y":s1_full.y,
})
#landuse back to uint8 (more data efficient) and in Sentinel 1
landuse_reprojected = landuse_reprojected.astype(np.uint8)
s1_full['landuse'] = landuse_reprojected.isel(band = 0) #drop the band

#%% Write landuse and S1 to new NetCDF
s1_full.to_netcdf('data/s0_OpenEO/S0_zwalm_landuse.nc', mode = 'w')