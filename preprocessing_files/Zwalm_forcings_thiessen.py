import os
import pickle
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geovoronoi import coords_to_points, voronoi_regions_from_coords
from geovoronoi.plotting import (plot_voronoi_polys_with_points_in_area,
                                 subplot_for_map)
from shapely import geometry
from shapely.ops import voronoi_diagram
from itertools import combinations
import hvplot
import hvplot.pandas
pad = Path(os.getcwd())
if pad.name != "Python":
    pad_correct = Path("../../Python")
    os.chdir(pad_correct)

#pyright: reportUnboundVariable=false

#http://daad.wb.tu-harburg.de/?id=279

#Read in the pickled files
#with open("employee_info.pickle", "rb") as file:
#   loaded_dict = pickle.load(file)
pickled_folder = Path("data/Zwalm_data/pywaterinfo_output")
P_dict = pickle.load(open(pickled_folder/"P_dict.pickle", "rb"))
P_info_dict = pickle.load(open(pickled_folder/"P_info_dict.pickle", "rb"))
EP_dict = pickle.load(open(pickled_folder/"EP_dict.pickle", "rb"))
EP_info_dict = pickle.load(open(pickled_folder/"EP_info_dict.pickle", "rb"))

################################
# Functions to use in this script
################################

def custom_thiessen_polygons(gdf_info, box_shape, gdf_catchment):
    """Process outpout of geovoronoi so that only shape of catchment is considered

    Parameters
    ----------
    gdf_info: geopandas.GeoDataFrame
        Each row is a station, 3 columns must be present:
        - name: 
        - station_name
        - geometry: contains location as point (shapely)
    gdf_catchment: geopandas.GeoDataFrame
        must contain 'geometry' columns wit the polygon shape of the catchment

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

##########################################
#geopandas dataframes: converting or making 
##########################################

keys_P = list(P_info_dict.keys())
P_info_pd = P_info_dict[keys_P[0]]
for i in np.arange(1,len(keys_P)):
    pddf = P_info_dict[keys_P[i]]
    P_info_pd = pd.concat([P_info_pd,pddf], ignore_index= True)
    gdf_P_info = gpd.GeoDataFrame(
        P_info_pd, 
        geometry=gpd.points_from_xy(
            P_info_pd['station_local_x'],
            P_info_pd['station_local_y'],
            crs = "EPSG:31370"
        )
        )# type: ignore 
    gdf_P_info = gdf_P_info.set_crs('EPSG:31370')
gdf_P_info['name'] = keys_P
#copy paste: exactly the same for EP as for P
keys_EP = list(EP_info_dict.keys())
EP_info_pd = EP_info_dict[keys_EP[0]]
for i in np.arange(1,len(keys_EP)):
    pddf = EP_info_dict[keys_EP[i]]
    EP_info_pd = pd.concat([EP_info_pd,pddf], ignore_index= True)
    gdf_EP_info = gpd.GeoDataFrame(
        EP_info_pd, 
        geometry=gpd.points_from_xy(
            EP_info_pd['station_local_x'],
            EP_info_pd['station_local_y'],
            crs = "EPSG:31370"
        )
        )# type: ignore 
gdf_EP_info['name'] = keys_EP

#Data on Zwalm: use epsg 31370 for distance calculation
zwalm_lambert = gpd.read_file(Path("data\Zwalm_shape\zwalm_shapefile_emma_31370.shp"))
zwalm_lambert_centroid = zwalm_lambert['geometry'].centroid
zwalm_area_shape = zwalm_lambert.loc[0,'geometry']

#make rectangle around the Zwalm catchment
zwalm_lambert['boundary'] = zwalm_lambert['geometry'].envelope
xx, yy = zwalm_lambert['boundary'].values[0].exterior.coords.xy

#make rectangle that encompasses al the points
local_x = gdf_P_info['station_local_x'].values.astype(np.float32)
xmin = min([np.min(local_x),np.min(xx)])
xmax = max([np.max(local_x),np.max(xx)])
local_y = gdf_P_info['station_local_y'].values.astype(np.float32)
ymin = min([np.min(local_y),np.min(yy)])
ymax = max([np.max(local_y),np.max(yy)])
box_shape = geometry.box(xmin,ymin,xmax,ymax)

#############################
# Thiessen for P
###############################

gdf_P_thiessen = gdf_P_info[['name','station_name','geometry']]#type:ignore
#idea: make 2 dictionaries
# 1) a dictionary that links each set combination to a number (the key)
# 2) a dictionary that has the correct geopandas dataframe with thiessen polygons for each combination

# 1) make dictionary of alll combinations
n = len(gdf_P_thiessen)
counter = 0
combinations_dict = {}
for i in np.arange(2,n+1):
    #start from 2: we only need combinations of a minimum of 2 stations!
    comb = list(combinations(gdf_P_thiessen['name'].to_list(),i))
    for j in comb: 
        combinations_dict[counter] = set(j)
        counter = counter + 1

# 2) make dictionary of geopandas dataframes
combinations_gdf_dict = {}
m = len(combinations_dict)
for i in range(m):
    station_list = list(combinations_dict[i])
    for index, j in enumerate(station_list):
        if index == 0:
            gdf_temp = gdf_P_thiessen[gdf_P_thiessen['name'] == j][
                ['station_name','name','geometry']
            ]
        elif index > 0:
            gdf_temp = gpd.GeoDataFrame(
                pd.concat([
                    gdf_temp,#type:ignore
                    gdf_P_thiessen[gdf_P_thiessen['name'] == j][
                        ['station_name','name','geometry']
                    ]
                ], ignore_index = True)
            )
    gdf_temp_thiessen = custom_thiessen_polygons(gdf_temp, box_shape, zwalm_lambert)
    gdf_temp_thiessen.plot()
    combinations_gdf_dict[i] = gdf_temp_thiessen

#check where nans occur in the datasets!
nstations = len(P_dict)
P_df_all = P_dict[keys_P[0]][['Timestamp','Value']]
P_df_all  = P_df_all.rename(columns = {'Value':keys_P[0]})
for i in np.arange(1,nstations):
    df_temp = P_dict[keys_P[i]][['Timestamp','Value']]
    df_temp = df_temp.rename(columns = {'Value':keys_P[i]})
    P_df_all = P_df_all.merge(df_temp, on = 'Timestamp')
df_no_time = P_df_all.drop('Timestamp', axis = 1)
df_bool = df_no_time.apply(lambda x: ~np.isnan(x))
def column_selector(row, indexes):
    indexes = list(indexes)
    locs = np.where(row.values)[0].tolist()
    names_list = list(indexes[i] for i in locs)
    return set(names_list)
P_df_all['nonan_station_sets'] = df_bool.apply(lambda x: column_selector(x, x.index), axis = 1)
P_df_all['#_nonan_stations'] = P_df_all['nonan_station_sets'].apply(lambda x: len(x))
P_df_all.plot(x= 'Timestamp', y='#_nonan_stations')

#find correct gdf_to accompany each timepoint
#set the index of combinations_gdf_dict as new column
n_comb = len(combinations_dict)
def give_index_for_gdf(set):
    return int(
        np.where(
            np.repeat(set, n_comb) == [combinations_dict[i] for i in range(len(combinations_dict))]
        )[0]
    )
P_df_all['gdf_index'] = P_df_all['nonan_station_sets'].apply(lambda x:give_index_for_gdf(x))

#now apply the calculated correction factors
def apply_correction_factors(row):


    return 3



#int(np.where(np.repeat(test_set, len(combinations_dict)) == [combinations_dict[i] for i in range(len(combinations_dict))])[0])

############################
# Thiessen for EP: analogous
############################
