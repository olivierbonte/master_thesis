# Convert shapefile emma with subbasins to 1 shapefile for Zwalm

import geopandas as gpd
from pathlib import Path
import os
import zipfile
pad = Path(os.getcwd())
if pad.name == "preprocessing_files":
    pad_correct = pad.parent
    os.chdir(pad_correct)
# os.system('zenodo_get 10.5281/zenodo.7688784')
os.system('zenodo_get 10.5281/zenodo.7971288')
with zipfile.ZipFile("data_github.zip", 'r') as zip_ref:
    zip_ref.extractall('data_github')
with zipfile.ZipFile("data.zip", 'r') as zip_ref:
    zip_ref.extractall('data')
zwalm_gpd_subbasins = gpd.read_file("data_github/OS266.shp")
circumference_zwalm = zwalm_gpd_subbasins.unary_union  # type:ignore
d = {'PolygonId': 15, 'Area': circumference_zwalm.area,
     'Subbasin': 0, 'geometry': circumference_zwalm}
zwalm_gpd = zwalm_gpd_subbasins.append(d, ignore_index=True)  # type:ignore
zwalm_gpd.crs = 31370  # set crs: Lambert72
zwalm_shape_epsg31370 = zwalm_gpd.iloc[[len(zwalm_gpd) - 1]]
if not os.path.exists('data/Zwalm_shape'):
    os.makedirs('data/Zwalm_shape')
zwalm_shape_epsg31370.to_file(
    Path(r"data/Zwalm_shape/zwalm_shapefile_emma_31370.shp"))
# To WGS 84
zwalm_gpd['geometry'] = zwalm_gpd['geometry'].to_crs(
    epsg=4326)  # change from 31370
zwalm_shape_epsg4326 = zwalm_gpd.iloc[[len(zwalm_gpd) - 1]]
zwalm_shape_epsg4326.to_file(
    Path(r"data/Zwalm_shape/zwalm_shapefile_emma.shp"))
