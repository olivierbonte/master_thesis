import pandas as pd
from pathlib import Path
import datetime
import numpy as np
import os
import geopandas as gpd
import shapely
pad = Path(os.getcwd())
if pad.name != "Python":
    pad_correct = Path("../../Python")
    os.chdir(pad_correct)


##########################################################
# Original processing: only 1 measurement point considered
##########################################################

dateparse = lambda x: datetime.datetime.strptime(x, "%d/%m/%Y %H:%M:%S")
data_p = pd.read_csv(Path("data\Zwalm_data\MaarkeKerkem_Neerslag_1h_geaccumuleerd.csv"), parse_dates = ['Datum'],
date_parser = dateparse, dtype = np.float32)
data_ep = pd.read_csv(Path("data\Zwalm_data\Liedekerke_ME_Potential_evapotranspiration_1h_geaccumuleerd.csv"), parse_dates = ['Date'],
date_parser = dateparse, dtype = np.float32)
data_q = pd.read_csv(Path("data\Zwalm_data\OS266_L06_342_Afvoer_hourly_reprocessed.csv"), parse_dates = ['Date'],
date_parser = dateparse, dtype = np.float32)

# Selecting ot time overlap between datasets
start_dates = [data_p['Datum'][0], data_ep['Date'][0], data_q['Date'][0]]
start_date = max(start_dates)
end_dates = [data_p['Datum'].iloc[-1], data_ep['Date'].iloc[-1], data_q['Date'].iloc[-1]]
end_date = min(end_dates)


#Precipitation: filter on time range and replace nan by zero
data_p = data_p.loc[(data_p['Datum'] >= start_date) & (data_p['Datum'] <= end_date)]
nan_bool_p = np.isnan(data_p['Neerslag'])
data_p.loc[nan_bool_p, 'Neerslag'] = 0

# Evaportaion: filter on time replace nan by zero
data_ep = data_ep.loc[(data_ep['Date'] >= start_date) & (data_ep['Date'] <= end_date)]
nan_bool_ep = np.isnan(data_ep['ET'])
data_ep.loc[nan_bool_ep, 'ET'] = 0

# Flow: only time filtering
data_q = data_q.loc[(data_q['Date'] >= start_date) & (data_q['Date'] <= end_date)]

# Merge into 1 table
data_p = data_p.rename(columns= {'Datum':'Time','Neerslag':'P'})
data_ep = data_ep.rename(columns= {'Date':'Time','ET':'EP'})
data_q = data_q.rename(columns= {'Date':'Time'})
data_zwalm = data_p.merge(data_ep, on = 'Time', how = 'left')
data_zwalm = data_zwalm.merge(data_q, on  = 'Time', how = 'left')

#Adapt so that 1 value per hour: where multiples occur, interpolation is done!
#Also only use full days: so go from 00:00 to 23:00
bool_00 = data_zwalm['Time'].dt.hour == 00
index_00 = np.argwhere(bool_00[:][:].to_list())
bool_23 = data_zwalm['Time'].dt.hour == 23
index_23 = np.argwhere(bool_23[:][:].to_list())
data_zwalm = data_zwalm[int(index_00[0]):int(index_23[-1])+1]

start_date_00 = data_zwalm['Time'].iloc[0]
end_date_23 = data_zwalm['Time'].iloc[-1]

timeseries = pd.date_range(start= start_date_00, end = end_date_23, freq= 'H')
time_df = pd.DataFrame({'Time':timeseries})
unique_times, index_unique = np.unique(data_zwalm['Time'], return_index= True) #select unique timestamps
data_zwalm_unique = data_zwalm.iloc[index_unique,:]
data_zwalm_hourly = time_df.merge(data_zwalm_unique, on = 'Time', how = 'left')
#interpolation where it had to fill in with Nan for P and EP
indeces_filled = np.argwhere(np.isnan(data_zwalm_hourly['P'].to_numpy()))
for i in indeces_filled:
    data_zwalm_hourly['P'].iloc[i] = 1/2*(data_zwalm_hourly['P'].iloc[i+1].values
     +  data_zwalm_hourly['P'].iloc[i-1].values)
    data_zwalm_hourly['EP'].iloc[i] = 1/2*( data_zwalm_hourly['EP'].iloc[i+1].values
     +  data_zwalm_hourly['EP'].iloc[i-1].values)

data_zwalm_hourly.to_csv('data/Zwalm_data/zwalm_forcings_flow.csv', index = False)

#also make a dataset where flow is aggregated on a daily basis
#timeseries_daily = pd.date_range(start = start_date_00, end = end_date_23, freq = 'D')
flow_zalm_hourly = data_zwalm_hourly[['Time','Flow']]
flow_zalm_hourly = flow_zalm_hourly.set_index('Time')
def custom_nanmean(arraylike):
    return np.nanmean(arraylike)
flow_zwalm_daily = flow_zalm_hourly.resample('1D').apply(custom_nanmean)
flow_zwalm_daily = flow_zwalm_daily.reset_index()
flow_zwalm_daily.to_csv("data/Zwalm_data/zwalm_flow_daily.csv", index = False)

###########################################################
# Extended processing: multiple measuring points considered
###########################################################
## Step 1: Inverse Distance Weighing
list_longitude_WGS84 = []
list_latitude_WGS84 = []
list_station_names = []
measurements_file = Path("data\Zwalm_data\waterinfo_measurements_rain")
for filename in os.listdir(measurements_file):
    if filename[-3:] == 'csv':
        #print(filename)
        info_station = pd.read_csv(measurements_file/filename, sep = ';', 
        nrows = 4, header=None, index_col= 0)
        latitude_WGS84 = info_station.loc[['#station_latitude']].values.flatten()
        list_latitude_WGS84.append(latitude_WGS84.astype(dtype = np.float64).tolist()[0])
        longitude_WGS84 = info_station.loc[['#station_longitude']].values.flatten()
        list_longitude_WGS84.append(longitude_WGS84.astype(dtype = np.float64).tolist()[0])
        list_station_names.append(info_station.loc[['#station_name']].values.tolist()[0])
df_stations = pd.DataFrame(
    {'Station': list_station_names,
     'Latitude':list_latitude_WGS84,
     'Longitude':list_longitude_WGS84})
gdf_stations = gpd.GeoDataFrame(
    df_stations, 
    geometry=gpd.points_from_xy(df_stations['Longitude'], df_stations['Latitude']))# type: ignore 
gdf_stations = gdf_stations.set_crs('EPSG:4326')
#testing for 1: extract coordinates! => later in a function
# info_station = pd.read_csv(measurements_file/"Zingem_P_Neerslag.csv", sep = ';', 
#     nrows = 4, header=None, index_col= 0)
# latitude_WGS84 = info_station.loc[['#station_latitude']].values.flatten()
# latitude_WGS84 = latitude_WGS84.astype(dtype = np.float64)
# longitude_WGS84 = info_station.loc[['#station_longitude']].values.flatten()
# longitude_WGS84 = longitude_WGS84.astype(dtype = np.float64)

#import shapefile of Zwalm: option 1 = EPSG 4326
#zwalm_WGS84 = gpd.read_file(Path("data\Zwalm_shape\zwalm_shapefile_emma.shp"))
#zwalm_WGS84_centroid = zwalm_WGS84['geometry'].centroid
#https://en.wikipedia.org/wiki/Centroid
#gdf_stations['distance'] = gdf_stations['geometry'].distance(zwalm_WGS84_centroid.iloc[0])

#import shapefile of Zwalm: option 2 = EPSG 31370 = BETTER for distances!!
zwalm_lambert = gpd.read_file(Path("data\Zwalm_shape\zwalm_shapefile_emma_31370.shp"))
zwalm_lambert_centroid = zwalm_lambert['geometry'].centroid
gdf_stations_lambert = gdf_stations.to_crs('EPSG:31370').copy()#type:ignore
gdf_stations_lambert['distance'] = gdf_stations_lambert['geometry'].distance(zwalm_lambert_centroid.iloc[0])#type: ignore

#inverse distance weighing
b = 1
#gdf_stations['IDW_factor_linear'] = gdf_stations['distance']**(-b)/sum(gdf_stations['distance']**(-b))
gdf_stations_lambert['IDW_factor_linear'] = gdf_stations_lambert['distance']**(-b)/sum(gdf_stations_lambert['distance']**(-b))
b =2
gdf_stations_lambert['IDW_factor_quadratic'] = gdf_stations_lambert['distance']**(-b)/sum(gdf_stations_lambert['distance']**(-b))
b = 3
gdf_stations_lambert['IDW_factor_cubic'] = gdf_stations_lambert['distance']**(-b)/sum(gdf_stations_lambert['distance']**(-b))
#factors -> csv
gdf_stations_lambert.to_csv(Path("data\Zwalm_data\zwalm_idw.csv"))


## Step 2: reading in the data

