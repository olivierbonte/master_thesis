import numpy as np
import pandas as pd
import geopandas as gpd
import os
import pickle
from pathlib import Path
pad = Path(os.getcwd())
if pad.name != "Python":
    pad_correct = Path("../../Python")
    os.chdir(pad_correct)

#Read in the pickled files
#with open("employee_info.pickle", "rb") as file:
#   loaded_dict = pickle.load(file)
pickled_folder = Path("data/Zwalm_data/pywaterinfo_output")
P_dict = pickle.load(open(pickled_folder/"P_dict.pickle", "rb"))
P_info_dict = pickle.load(open(pickled_folder/"P_info_dict.pickle", "rb"))
EP_dict = pickle.load(open(pickled_folder/"EP_dict.pickle", "rb"))
EP_info_dict = pickle.load(open(pickled_folder/"EP_info_dict.pickle", "rb"))


#converting to geopandas dataframes!
keys_P = list(P_info_dict.keys())
P_info_pd = P_info_dict[keys_P[0]]
for i in np.arange(1,len(keys_P)):
    pddf = P_info_dict[keys_P[i]]
    P_info_pd = pd.concat([P_info_pd,pddf], ignore_index= True)
    gdf_P_info = gdf_stations = gpd.GeoDataFrame(
        P_info_pd, 
        geometry=gpd.points_from_xy(
            P_info_pd['station_local_x'],
            P_info_pd['station_local_y'],
            crs = "EPSG:31370"
        )
        )# type: ignore 
    gdf_P_info = gdf_P_info.set_crs('EPSG:31370')
#copy paste: exactly the same for EP as for P
keys_EP = list(EP_info_dict.keys())
EP_info_pd = EP_info_dict[keys_EP[0]]
for i in np.arange(1,len(keys_EP)):
    pddf = EP_info_dict[keys_EP[i]]
    EP_info_pd = pd.concat([EP_info_pd,pddf], ignore_index= True)
    gdf_EP_info = gdf_stations = gpd.GeoDataFrame(
        EP_info_pd, 
        geometry=gpd.points_from_xy(
            EP_info_pd['station_local_x'],
            EP_info_pd['station_local_y'],
            crs = "EPSG:31370"
        )
        )# type: ignore 