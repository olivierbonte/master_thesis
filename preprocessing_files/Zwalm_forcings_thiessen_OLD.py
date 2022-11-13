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
# REMARK that geovoronoi is NOT anymore a defulat in the environment.yml, to run this script,
# install this package serepately via `pip install geovoronoi[plotting]` 

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
def mask_thiessen_polygons(gdf_info, region_pts, region_polys, gdf_catchment):
    """Process outpout of geovoronoi so that only shape of catchment is considered

    Parameters
    ----------
    gdf_info: geopandas.GeoDataFrame
        Each row is a station, 3 columns must be present:
        - name: 
        - station_name
        - geometry: contains location as point (shapely)

    region_pts: 
        output from voronoi_regions_from_coords, maps point to index of points in geometry
    region_polys:
        output from voronoi_regions_from_coords, polygons
    gdf_catchment: geopandas.GeoDataFrame
        must contain 'geometry' columns wit the polygon shape of the catchment

    Returns
    --------
    gdf_thiessen_catchment: geopandas.GeoDataFrame   
        geometry column containing the Thiessen polygons within the catchment, also 
        area and relative area included 
    """
    
    P_points = coords_to_points(gdf_info['geometry'])#type: ignore
    gdf_info = gdf_info.rename(columns= {'geometry':'location'})#type:ignore
    list_thiessen_polys = [None] * len(P_points)
    for i in range(len(region_polys)):
        index_info = region_pts[i][0] #indicates which point of P_points are related to the polyon
        list_thiessen_polys[index_info] = region_polys[i]
    gdf_thiessen = gdf_info.copy()
    gdf_thiessen['geometry'] = list_thiessen_polys
    gdf_thiessen['geometry'] = gdf_thiessen['geometry'].astype('geometry')
    gdf_thiessen.set_crs('EPSG:31370')
    gdf_thiessen_catchment = gdf_thiessen.overlay(gdf_catchment[['geometry']], how='intersection')#type:ignore
    gdf_thiessen_catchment['Area'] = gdf_thiessen_catchment.area
    gdf_thiessen_catchment['relative_area'] = gdf_thiessen_catchment['Area']/np.sum(gdf_thiessen_catchment['Area'])
    return gdf_thiessen_catchment

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
    gdf_thiessen['geometry'] = [region for region in vonoroi_shapely]#type:ignore
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

# FIRST: Voroni diagram for all points 
# 1) Voroni diagrams construction
P_points = coords_to_points(gdf_P_info['geometry'])#type: ignore
#poly = geometry.Polygon([[p.x, p.y] for p in gdf_P_info['geometry'].to_list()])
#P_points.append(coords_to_points(zwalm_lambert_centroid)[0])
region_polys, region_pts = voronoi_regions_from_coords(P_points, box_shape)#type:ignore
fig, ax = subplot_for_map()
plot_voronoi_polys_with_points_in_area(ax, box_shape, region_polys, P_points, region_pts)
plt.show()

# 2) Find the area of the polygon WITHIN the Zwalm catchment
# gdf_P_thiessen = gdf_P_info[['name','station_name','geometry']]#type:ignore
# gdf_P_thiessen = gdf_P_thiessen.rename(columns= {'geometry':'location'})#type:ignore
# list_thiessen_polys = [None] * len(P_points)
# for i in range(len(region_polys)):
#     index_info = region_pts[i][0] #indicates which point of P_points are related to the polyon
#     list_thiessen_polys[index_info] = region_polys[i]
# gdf_P_thiessen['geometry'] = list_thiessen_polys
# gdf_P_thiessen['geometry'] = gdf_P_thiessen['geometry'].astype('geometry')
# gdf_P_thiessen.set_crs('EPSG:31370')
# gdf_P_thiessen_zwalm = gdf_P_thiessen.overlay(zwalm_lambert[['geometry']], how='intersection')
# #Conclusie: met Thiessen polygonen kan je NIET Ronse meenemen in de tijdreeks!
# gdf_P_thiessen_zwalm['Area'] = gdf_P_thiessen_zwalm.area
# gdf_P_thiessen_zwalm['relative_area'] = gdf_P_thiessen_zwalm['Area']/np.sum(gdf_P_thiessen_zwalm['Area'])
gdf_P_thiessen = gdf_P_info[['name','station_name','geometry']]#type:ignore
gdf_P_thiessen_zwalm = mask_thiessen_polygons(gdf_P_thiessen, region_pts, region_polys, zwalm_lambert)

## SECOND: Thiesen polygons for when NOT all stations are present!
#test: a set of {Zingem, Elst} of rainfall!
#idea: make 2 dictionaries
# 1) a dictionary that links each set combination to a number (the key)
# 2) a dictionary that has the correct geopandas dataframe with thiessen polygons for each combination
testset = {'Zingem','Elst'}

#make dictionary of all combinations, including Ronse!
n = len(gdf_P_thiessen)
counter = 0
combinations_dict = {}
for i in np.arange(2,n):
    #start from 2: we only need combinations of a minimum of 2 stations!
    comb = list(combinations(gdf_P_thiessen['name'].to_list(),i))
    for j in comb: 
        combinations_dict[counter] = set(j)
        counter = counter + 1
#Custom for 2 points function
point1 = gdf_P_thiessen_zwalm['location'][0]
point2 = gdf_P_thiessen_zwalm['location'][1]
# def custom_thiessen(point1, point2, box):
#     #x_mid = (point1.x + point2.x)/2
#     #y_mid = (point1.x + point2.x)/2
#     list_ext_coords_box = list(box.exterior.coords)
#     ab = geometry.LineString([point1, point2])
#     l = box_shape.length
#     left = ab.parallel_offset(l, 'left')
#     right = ab.parallel_offset(l, 'right')
#     x_left = (left.boundary[0].x + left.boundary[1].x)/2
#     y_left = (left.boundary[0].y + left.boundary[1].y)/2
#     x_right = (right.boundary[0].x + right.boundary[1].x)/2
#     y_right = (right.boundary[0].y + right.boundary[1].y)/2
#     left_point = geometry.Point(x_left, y_left)
#     right_point = geometry.Point(x_right, y_right)
#     middelloodlijn = geometry.LineString(([left_point,right_point]))
#     #import pdb; pdb.set_trace()
#     dict = {
#         'geometry':[box]
#     }
#     gdf_test = gpd.GeoDataFrame(dict, crs = 'EPSG:31370')
#     gdf_test = gdf_test.append({'geometry':ab}, ignore_index = True)
#     gdf_test = gdf_test.append({'geometry':middelloodlijn}, ignore_index = True)
#     return gdf_test
# gdf_test = custom_thiessen(point1, point2, box_shape)
#splitted = shapely
#test shapely thiessen
points = geometry.MultiPoint([point1, point2])
vonoroi_shapely = voronoi_diagram(points, box_shape) ##!!WERKT VOOR 2 PUNTEN!!


#make dictionary of geopandas dataframes
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
    #if len(combinations_dict[i]) == 2:
        # points = geometry.MultiPoint(gdf_temp['geometry'])
        # vonoroi_shapely = voronoi_diagram(points, box_shape) ##!!WERKT VOOR 2 PUNTEN!!
        # gdf_temp = gdf_temp.rename(columns = {'geometry':'location'})#type:ignore
        # gdf_temp['geometry'] = [region for region in vonoroi_shapely]#type:ignore
        # gdf_temp['geometry'] = gdf_temp['geometry'].astype('geometry')
        # #import pdb; pdb.set_trace(),
        # gdf_temp.set_crs('EPSG:31370')
        # gdf_thiessen_catchment = gdf_temp.overlay(zwalm_lambert[['geometry']], how='intersection')#type:ignore
        # gdf_thiessen_catchment['Area'] = gdf_thiessen_catchment.area
        # gdf_thiessen_catchment['relative_area'] = gdf_thiessen_catchment['Area']/np.sum(gdf_thiessen_catchment['Area'])
        # gdf_thiessen_catchment.plot()
        # gdf_temp_thiessen =  gdf_thiessen_catchment
    gdf_temp_thiessen = custom_thiessen_polygons(gdf_temp, box_shape, zwalm_lambert)
    gdf_temp_thiessen.plot()
    # else:
    #     P_points = coords_to_points(gdf_temp['geometry'])
    #     region_polys, region_pts = voronoi_regions_from_coords(P_points, box_shape)#type:ignore
    #     fig, ax = subplot_for_map()
    #     plot_voronoi_polys_with_points_in_area(ax, box_shape, region_polys, P_points, region_pts)
    #     plt.show()
    #     gdf_temp_thiessen = mask_thiessen_polygons(gdf_temp, region_pts, region_polys, zwalm_lambert)
    combinations_gdf_dict[i] = gdf_temp_thiessen

#check where nans occur in the datasets!
nstations = len(P_dict)
P_df_all = P_dict[keys_P[0]][['Timestamp','Value']]
P_df_all  = P_df_all.rename(columns = {'Value':keys_P[0]})
for i in np.arange(1,nstations):
    df_temp = P_dict[keys_P[i]][['Timestamp','Value']]
    df_temp = df_temp.rename(columns = {'Value':keys_P[i]})
    P_df_all = P_df_all.merge(df_temp, on = 'Timestamp')
# list_nonan_stations = [None] * len(P_df_all)
# list_number_nonan_stations = [None] * len(P_df_all)
# for i in range(len(P_df_all)): #non-efficient... 
#     df_no_time = P_df_all.drop('Timestamp', axis = 1)
#     indexing_bool = ~np.isnan(df_no_time.iloc[i,:])
#     columns = df_no_time.columns
#     nonan_stations = columns[indexing_bool]
#     list_nonan_stations[i] = set(nonan_stations)
#     list_number_nonan_stations[i] = len(set(nonan_stations))
# P_df_all['nonan_station_sets'] = list_nonan_stations
# P_df_all['#_nonan_stations'] = list_number_nonan_stations

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

########################
# EP: evapotranspiration
########################
