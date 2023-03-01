import os
import pickle
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import geometry
from itertools import combinations
import hvplot
import hvplot.pandas
pad = Path(os.getcwd())
if pad.name == "preprocessing_files":
    pad_correct = pad.parent
    os.chdir(pad_correct)
from functions.pre_processing import custom_thiessen_polygons
#pyright: reportUnboundVariable=false
write = True
#http://daad.wb.tu-harburg.de/?id=279

#Read in the pickled files
#with open("employee_info.pickle", "rb") as file:
#   loaded_dict = pickle.load(file)
pickled_folder = Path("data/Zwalm_data/pywaterinfo_output")
P_dict = pickle.load(open(pickled_folder/"P_dict.pickle", "rb"))
P_info_dict = pickle.load(open(pickled_folder/"P_info_dict.pickle", "rb"))
EP_dict = pickle.load(open(pickled_folder/"EP_dict.pickle", "rb"))
EP_info_dict = pickle.load(open(pickled_folder/"EP_info_dict.pickle", "rb"))

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
if not os.path.exists('data/Zwalm_data/preprocess_output'):
    os.makedirs('data/Zwalm_data/preprocess_output')
gdf_P_info.to_pickle('data/Zwalm_data/preprocess_output/gdf_P_info.pkl')
gdf_EP_info.to_pickle('data/Zwalm_data/preprocess_output/gdf_EP_info.pkl')

#Data on Zwalm: use epsg 31370 for distance calculation
zwalm_lambert = gpd.read_file(Path("data\Zwalm_shape\zwalm_shapefile_emma_31370.shp"))
zwalm_lambert_centroid = zwalm_lambert['geometry'].centroid
zwalm_area_shape = zwalm_lambert.loc[0,'geometry']

#make rectangle around the Zwalm catchment
zwalm_lambert['boundary'] = zwalm_lambert['geometry'].envelope
xx, yy = zwalm_lambert['boundary'].values[0].exterior.coords.xy

#make rectangle that encompasses al the points
local_x = gdf_P_info['station_local_x'].values.astype(np.float32)
xmin = min([np.min(local_x),np.min(xx)])#type:ignore
xmax = max([np.max(local_x),np.max(xx)])#type:ignore
local_y = gdf_P_info['station_local_y'].values.astype(np.float32)
ymin = min([np.min(local_y),np.min(yy)])#type:ignore
ymax = max([np.max(local_y),np.max(yy)])#type:ignore
box_shape = geometry.box(xmin,ymin,xmax,ymax)

#############################
# Thiessen for P
###############################

gdf_P_thiessen = gdf_P_info[['name','station_name','geometry']]#type:ignore
#idea: make 2 dictionaries
# 1) a dictionary that links each set combination to a number (the key)
# 2) a dictionary that has the correct geopandas dataframe with thiessen polygons for each combination

# 1) make dictionary of all combinations
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
def give_index_for_gdf(set, combinations_dict):
    if len(set) > 1:
        index = int(
            np.where(
                np.repeat(set, n_comb) == [combinations_dict[i] for i in range(len(combinations_dict))]
            )[0]
        )
    else: #there is no corresponding index to combination if only 0 or 1 stations
        index = str(set) #can be used later! 
    return index
give_index_for_gdf_P = lambda x: give_index_for_gdf(x, combinations_dict)
P_df_all['gdf_index'] = P_df_all['nonan_station_sets'].apply(lambda x:give_index_for_gdf_P(x))

#now apply the calculated correction factors
def apply_correction_factors(row, comb_gdf_dict):
    gdf_index = row['gdf_index']
    if isinstance(gdf_index, int):
        gdf = comb_gdf_dict[gdf_index]
        gdf_name = gdf.set_index('name')
        value = 0
        for name in list(gdf_name.index):
            value = value + row[name]*gdf_name.loc[name].relative_area
    elif gdf_index == 'set()': #the case for 0 statoins
        value = np.nan
    else: # so 1 station
        current_name = gdf_index[2:-2] #cut of {} in string
        value = row[current_name] #here gdf_index is an set
    return value
apply_correction_factors_P = lambda x: apply_correction_factors(x, combinations_gdf_dict)
P_df_all['P_thiessen'] = P_df_all.apply(lambda x: apply_correction_factors_P(x), axis = 1)
P_df_all.hvplot(x = 'Timestamp', y = ['P_thiessen','Zingem','Maarke-Kerkem','Elst','Ronse'])
#int(np.where(np.repeat(test_set, len(combinations_dict)) == [combinations_dict[i] for i in range(len(combinations_dict))])[0])
if write:
    P_df_all.to_csv(Path('data/Zwalm_data/preprocess_output/zwalm_p_thiessen.csv'))
    P_df_all.to_pickle(Path('data/Zwalm_data/preprocess_output/zwalm_p_thiessen.pkl'))
    with open('data/Zwalm_data/preprocess_output/all_p_polygon_combinations.pkl', 'wb') as handle:
        pickle.dump(combinations_gdf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


############################
# Thiessen for EP: analogous
############################
#completely analogous, to P (just copy paste as for now)


gdf_EP_thiessen = gdf_EP_info[['name','station_name','geometry']]#type:ignore

#make rectangle that encompasses al the points
local_x = gdf_EP_info['station_local_x'].values.astype(np.float32)
xmin = min([np.min(local_x),np.min(xx)])#type:ignore
xmax = max([np.max(local_x),np.max(xx)])#type:ignore
local_y = gdf_EP_info['station_local_y'].values.astype(np.float32)
ymin = min([np.min(local_y),np.min(yy)])#type:ignore
ymax = max([np.max(local_y),np.max(yy)])#type:ignore
box_shape_EP = geometry.box(xmin,ymin,xmax,ymax)
# 1) make dictionary of all combinations
n = len(gdf_EP_thiessen)
counter = 0
combinations_dict_EP = {}
for i in np.arange(2,n+1):
    #start from 2: we only need combinations of a minimum of 2 stations!
    comb = list(combinations(gdf_EP_thiessen['name'].to_list(),i))
    for j in comb: 
        combinations_dict_EP[counter] = set(j)
        counter = counter + 1

# 2) make dictionary of geopandas dataframes
combinations_gdf_dict_EP = {}
m = len(combinations_dict_EP)
for i in range(m):
    station_list = list(combinations_dict_EP[i])
    for index, j in enumerate(station_list):
        if index == 0:
            gdf_temp = gdf_EP_thiessen[gdf_EP_thiessen['name'] == j][
                ['station_name','name','geometry']
            ]
        elif index > 0:
            gdf_temp = gpd.GeoDataFrame(
                pd.concat([
                    gdf_temp,#type:ignore
                    gdf_EP_thiessen[gdf_EP_thiessen['name'] == j][
                        ['station_name','name','geometry']
                    ]
                ], ignore_index = True)
            )
    gdf_temp_thiessen = custom_thiessen_polygons(gdf_temp, box_shape_EP, zwalm_lambert)
    gdf_temp_thiessen.plot()
    combinations_gdf_dict_EP[i] = gdf_temp_thiessen

#check where nans occur in the datasets!
nstations = len(EP_dict)
EP_df_all = EP_dict[keys_EP[0]][['Timestamp','Value']]
EP_df_all  = EP_df_all.rename(columns = {'Value':keys_EP[0]})
for i in np.arange(1,nstations):
    df_temp = EP_dict[keys_EP[i]][['Timestamp','Value']]
    df_temp = df_temp.rename(columns = {'Value':keys_EP[i]})
    EP_df_all = EP_df_all.merge(df_temp, on = 'Timestamp')
df_no_time = EP_df_all.drop('Timestamp', axis = 1)
df_bool = df_no_time.apply(lambda x: ~np.isnan(x))
EP_df_all['nonan_station_sets'] = df_bool.apply(lambda x: column_selector(x, x.index), axis = 1)
EP_df_all['#_nonan_stations'] = EP_df_all['nonan_station_sets'].apply(lambda x: len(x))
EP_df_all.hvplot(x= 'Timestamp', y='#_nonan_stations')

#find correct gdf_to accompany each timepoint
#set the index of combinations_gdf_dict as new column
n_comb = len(combinations_dict_EP)
give_index_for_gdf_EP = lambda x: give_index_for_gdf(x, combinations_dict_EP)
EP_df_all['gdf_index'] = EP_df_all['nonan_station_sets'].apply(lambda x:give_index_for_gdf_EP(x))
#now apply the calculated correction factors
apply_correction_factors_EP = lambda x:apply_correction_factors(x, combinations_gdf_dict_EP)
EP_df_all['EP_thiessen'] = EP_df_all.apply(lambda x: apply_correction_factors_EP(x), axis = 1)

#PROBLEM: THERE ARE STILL NAN VALUES PRESENT!
print(sum(np.isnan(EP_df_all['EP_thiessen'])))
EP_df_all['Thiessen_Nan'] = np.isnan(EP_df_all['EP_thiessen'])
#Prepare data for Nan filtering
EP_df_all['ymd'] = EP_df_all['Timestamp'].apply(lambda x: pd.Timestamp(year = x.year, month = x.month, day = x.day))
#EP_df_all['mdh'] =  P_df_all['Timestamp'].apply(lambda x: pd.Timestamp(month = x.month, day = x.day, hour = x.hour))

#UPDATE HANS: the method below is NOT GOOD! Does not take into account daily variation!  
#Nan Filter 1: Take daily average when Nan is present
# daily_mean = EP_df_all[['EP_thiessen','ymd']].groupby('ymd').mean()
# daily_mean = daily_mean.rename(columns = {'EP_thiessen':'EP_thiessen_daily_mean'})
# EP_df_all = EP_df_all.set_index('ymd').join(daily_mean, how = 'left').reset_index()
# EP_df_all['EP_thiessen'] = EP_df_all['EP_thiessen'].fillna(EP_df_all['EP_thiessen_daily_mean'])
# EP_df_all = EP_df_all.set_index('Timestamp').reset_index()

#Nan filter 2: If no value for an entire day, take average of other years at this time stamp!
#This is done for every timestamp (so for every hour of the year, an average value)
EP_df_all['month'] = EP_df_all['Timestamp'].dt.month
EP_df_all['day'] = EP_df_all['Timestamp'].dt.day
EP_df_all['hour'] = EP_df_all['Timestamp'].dt.hour
average_year = EP_df_all[['EP_thiessen','month','day','hour']].groupby(
    ['month','day','hour']).mean()
average_year = average_year.rename(columns = {'EP_thiessen':'EP_thiessen_ave_yearly'})
EP_df_all = EP_df_all.set_index(['month','day','hour']).join(average_year, how = 'left').reset_index()
EP_df_all['EP_thiessen_filled'] = EP_df_all['EP_thiessen']
EP_df_all['EP_thiessen_filled'] = EP_df_all['EP_thiessen_filled'].fillna(EP_df_all['EP_thiessen_ave_yearly'])
Nan_days = EP_df_all[EP_df_all['Thiessen_Nan'] == True]['ymd'].unique()
for day in Nan_days:
    EP_df_temp = EP_df_all[EP_df_all['ymd'] == day]
    nan_moments_bool = np.isnan(EP_df_temp['EP_thiessen'])
    nan_moments_indexes = nan_moments_bool[nan_moments_bool == True].index
    nonan_hours = EP_df_temp.loc[~nan_moments_bool,'hour']
    nan_day = pd.Timestamp(day).day
    nan_month = pd.Timestamp(day).month
    if sum(nan_moments_bool) < 24: #so not the entire day Nan!
        mean_nan_period = np.mean(EP_df_temp.loc[~nan_moments_bool,'EP_thiessen'])
        EP_df_other_years = EP_df_all[EP_df_all['day'] == nan_day]
        EP_df_other_years = EP_df_other_years[EP_df_other_years['month'] == nan_month]
        EP_df_other_years = EP_df_other_years[EP_df_other_years['hour'].isin(nonan_hours)]
        mean_other_years = np.mean(EP_df_other_years['EP_thiessen'])
        #'EP_thiessen_filled#only resscale the acutal Nan moments!
        EP_df_all.loc[nan_moments_indexes, 'EP_thiessen_filled'] = EP_df_all.loc[nan_moments_indexes, 'EP_thiessen_filled']*mean_nan_period/mean_other_years

#drop EP_thiessen
EP_df_all = EP_df_all.drop('EP_thiessen', axis = 1)
#repname EP_thiessen_filled to EP_thiessen
EP_df_all = EP_df_all.rename(columns = {'EP_thiessen_filled':'EP_thiessen'})
EP_df_all = EP_df_all.set_index('Timestamp').sort_index().reset_index() #Get chronologial order back!
print(sum(np.isnan(EP_df_all['EP_thiessen'])))
EP_df_all[['Timestamp','EP_thiessen','Thiessen_Nan']].hvplot(x= 'Timestamp')
EP_df_all = EP_df_all.drop(['ymd','month','day','hour'], axis = 1)
#EP_df_all.plot('Timestamp',['Liedekerke','Waregem','EP_thiessen'])
if write:
    EP_df_all.to_csv(Path('data/Zwalm_data/preprocess_output/zwalm_ep_thiessen.csv'))
    EP_df_all.to_pickle(Path('data/Zwalm_data/preprocess_output/zwalm_ep_thiessen.pkl'))
    with open('data/Zwalm_data/preprocess_output/all_ep_polygon_combinations.pkl', 'wb') as handle:
        pickle.dump(combinations_gdf_dict_EP, handle, protocol=pickle.HIGHEST_PROTOCOL)