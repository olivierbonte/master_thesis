
import rasterio
from pathlib import Path
import os
pad = Path(os.getcwd())
if pad.name != "Python":
    pad_correct = Path("../../Python")
    os.chdir(pad_correct)
from functions.pre_processing import mask_tiff_with_shape

#uitgevoerd in Lambert72
landuse = rasterio.open(Path("data\Zwalm_bodembedekking\BBK5_18\GeoTIFF\BBK5_18_Kbl30.tif"))
mask_tiff_with_shape(raster_rio_tiff = landuse, 
    filepath_shapefile = Path("data\Zwalm_shape\OS266.shp"),
    filepath_out= Path("temp/data/bodembekking_masked_test.tif"),
    nodata = 255) #use 255 since this does not occur
