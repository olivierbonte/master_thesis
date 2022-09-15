# Write code to allow preprocessing of the data

###################
# Zwalm: processing band 110
##################

import os
from pathlib import Path
from functions.pre_processing import pre_processing_pipeline
from joblib import Parallel, delayed
import glob 


filepath_shapefile = Path('data/Zwalm_shape/zwalm_shapefile_emma.shp')
files = Path('data/g0_020m').glob('*_110.nc')
n_fils = len(list(files))
files = Path('data/g0_020m').glob('*_110.nc')
output_dir = 'data/g0_020m_Zwalm'

#parallele versie
def pre_processing_parallel(filepath_shapefile, file, n_fils, i, output_dir, overwrite):
    #remove files to avoid problems
    message = "preprocessing of file " + str(i) + " out of " + str(n_fils)
    print(message)
    output_file = file.name[0:-3] + "_Zwalm.nc"
    filepath_nc_processed =  output_dir + '/' + output_file
    if os.path.exists(filepath_nc_processed):
        if overwrite:
            os.remove(filepath_nc_processed)
            pre_processing_pipeline(
                filepath_nc_raw = file,
                filepath_shapefile = filepath_shapefile,
                filepath_nc_processed =  output_dir + '/' + output_file,
                filepath_temp_data= 'data/temp',
                epsg = 4326,
                return_bool= False
            )  
    else:
        pre_processing_pipeline(
            filepath_nc_raw = file,
            filepath_shapefile = filepath_shapefile,
            filepath_nc_processed =  output_dir + '/' + output_file,
            filepath_temp_data= 'data/temp',
            epsg = 4326,
            return_bool= False
        ) 
    if os.path.exists("data/temp/" + file.name + "_masked.tiff"):
        os.remove("data/temp/" + file.name + "_masked.tiff")
    if os.path.exists("data/temp/" + file.name + "_raw.tiff"):
        os.remove("data/temp/" + file.name + "_raw.tiff")
    if os.path.exists("data/temp" + file.name  + "_masked.nc"):
        os.remove("data/temp/" + file.name + "_masked.nc")


Parallel(n_jobs= -1)(
    delayed(pre_processing_parallel)(filepath_shapefile, file, n_fils, i, output_dir, False) for i, file in enumerate(files)
)

# files = glob.glob('data/temp/')
# for f in files:
#     os.remove(f)
# os.remove('data/temp/')

 ## KLAD:  uncomment or comment with ctrl + /
#i = 1
#for file in files: 
#    #remove files to avoid problems
#    if os.path.exists("data/temp/masked.tiff"):
#       os.remove("data/temp/masked.tiff")
#    if os.path.exists("data/temp/raw.tiff"):
#        os.remove("data/temp/raw.tiff")
#    if os.path.exists("data/temp/masked.nc"):
#        os.remove("data/temp/masked.nc")
#     message = "preprocessing of file " + str(i) + " out of " + str(n_fils)
#     print(message)
#     i += 1
#     output_file = file.name[0:-3] + "_Zwalm.nc"
#     filepath_nc_processed =  output_dir + '/' + output_file
#     if os.path.exists(filepath_nc_processed):
#         os.remove(filepath_nc_processed)
#     pre_processing_pipeline(
#        filepath_nc_raw = file,
#        filepath_shapefile = filepath_shapefile,
#        filepath_nc_processed =  output_dir + '/' + output_file,
#        filepath_temp_data= 'data/temp',
#        epsg = 4326,
#        return_bool= False
#     )
#     if i == 1:
#         break 