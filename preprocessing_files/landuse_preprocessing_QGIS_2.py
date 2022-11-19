import sys
print (sys.version)
import processing 
import os
#Note that the script can only be run in the python console of QGIS!

pad = os.getcwd()
print(pad)
if not pad.endswith("Python"):
    pad_correct = "C:/Users/olivi/documents/Masterthesis_lokaal/Python"
    #Change the above path to local installation of main Python folder of code 
    os.chdir(pad_correct)
    
# Idea here is the direct conversion from EPSG 31370 to
# resolution AND extent of the Sentinel1 data
# remove existing files
project = QgsProject.instance()
if os.path.exists('data/Zwalm_bodembedekking/QGIS_project/Landuse_directly_to_Sentinel_resolution.tif'):
    to_be_deleted = project.mapLayersByName('Landuse_directly_to_Sentinel_resolution')[0]
    project.removeMapLayer(to_be_deleted.id())
    os.remove('data/Zwalm_bodembedekking/QGIS_project/Landuse_directly_to_Sentinel_resolution.tif')
processing.runAndLoadResults("gdal:warpreproject", 
{'INPUT':'data/Zwalm_bodembedekking/wetransfer_landgebruik_2022-11-07_0921/Landuse_Vlaanderen_Wallonie_final.sdat',
'SOURCE_CRS':QgsCoordinateReferenceSystem('EPSG:31370'),
'TARGET_CRS':QgsCoordinateReferenceSystem('EPSG:4326'),
'RESAMPLING':0,'NODATA':None,'TARGET_RESOLUTION':None,
#Target resolution oculd be calculated as 0.2/1008 (explained by Hans)
'OPTIONS':'','DATA_TYPE':5, #to set to int
'TARGET_EXTENT':'3.667460317,3.838293651,50.763095238,50.903571429 [EPSG:4326]',
#Target exent from a NetCDF of Sentinel Data
'TARGET_EXTENT_CRS':QgsCoordinateReferenceSystem('EPSG:4326'),
'MULTITHREADING':False,'EXTRA':'-ts 1722 1416',
#twice the number of the cells in original Sentinel Grid: this is 0.1/1008° res
'OUTPUT':'data/Zwalm_bodembedekking/QGIS_project/Landuse_directly_to_Sentinel_resolution.tif'})
