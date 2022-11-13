import pandas as pd
from pathlib import Path
import datetime
import numpy as np
import os
import geopandas as gpd

pad = Path(os.getcwd())
if pad.name != "Python":
    pad_correct = Path("../../Python")
    os.chdir(pad_correct)

################
# Functions used
################

def find_00_23(ds):
    """
    Find a starting data at 00:00 and end date at 00:23 for a pandas.Series
    containing time stamps

    Parameters
    ----------
    ds: pandas.Series   
        the timestamps

    Returns
    -------
    start_date_00: Timestamp
    end_date_23: Timestamp
    """
    bool_00 = ds.dt.hour == 00
    index_00 = np.argwhere(bool_00[:][:].to_list())
    bool_23 = ds.dt.hour == 23
    index_23 = np.argwhere(bool_23[:][:].to_list())
    ds= ds[int(index_00[0]):int(index_23[-1])+1]
    start_date_00 = ds.iloc[0]
    end_date_23 = ds.iloc[-1]
    return start_date_00, end_date_23

##########################################################
# Original processing: only 1 measurement point considered
##########################################################

dateparse = lambda x: datetime.datetime.strptime(x, "%d/%m/%Y %H:%M:%S")
data_p = pd.read_csv(Path("data\Zwalm_data\data_Jarne\MaarkeKerkem_Neerslag_1h_geaccumuleerd.csv"), parse_dates = ['Datum'],
date_parser = dateparse, dtype = np.float32)
data_ep = pd.read_csv(Path("data\Zwalm_data\data_Jarne\Liedekerke_ME_Potential_evapotranspiration_1h_geaccumuleerd.csv"), parse_dates = ['Date'],
date_parser = dateparse, dtype = np.float32)
data_q = pd.read_csv(Path("data\Zwalm_data\data_Jarne\OS266_L06_342_Afvoer_hourly_reprocessed.csv"), parse_dates = ['Date'],
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
start_date_00, end_date_23 = find_00_23(data_zwalm['Time'])
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

data_zwalm_hourly.to_csv('data/Zwalm_data/preprocess_output/zwalm_forcings_flow.csv', index = False)

#also make a dataset where flow is aggregated on a daily basis
#timeseries_daily = pd.date_range(start = start_date_00, end = end_date_23, freq = 'D')
flow_zalm_hourly = data_zwalm_hourly[['Time','Flow']]
flow_zalm_hourly = flow_zalm_hourly.set_index('Time')
def custom_nanmean(arraylike):
    return np.nanmean(arraylike)
flow_zwalm_daily = flow_zalm_hourly.resample('1D').apply(custom_nanmean)
flow_zwalm_daily = flow_zwalm_daily.reset_index()
flow_zwalm_daily.to_csv("data/Zwalm_data/preprocess_output/zwalm_flow_daily.csv", index = False)

###########################################################
# Extended processing: multiple measuring points considered
###########################################################

## STEP 1: INVERSE DISTANCE WEIGHING
list_longitude_WGS84 = []
list_latitude_WGS84 = []
list_station_names = []
measurements_file = Path("data\Zwalm_data\waterinfo_measurements_rain")
for filename in os.listdir(measurements_file):
    #if filename[-3:] == 'csv':
    #print(filename)
    info_station = pd.read_csv(measurements_file/filename, sep = ';', 
    nrows = 4, header=None, index_col= 0)
    latitude_WGS84 = info_station.loc[['#station_latitude']].values.flatten()
    list_latitude_WGS84.append(latitude_WGS84.astype(dtype = np.float64).tolist()[0])
    longitude_WGS84 = info_station.loc[['#station_longitude']].values.flatten()
    list_longitude_WGS84.append(longitude_WGS84.astype(dtype = np.float64).tolist()[0])
    #list_station_names.append(info_station.loc[['#station_name']].values.tolist()[0])
    list_station_names.append(info_station.loc['#station_name',1])
df_stations = pd.DataFrame(
    {'Station': list_station_names,
     'Latitude':list_latitude_WGS84,
     'Longitude':list_longitude_WGS84})
gdf_stations = gpd.GeoDataFrame(
    df_stations, 
    geometry=gpd.points_from_xy(df_stations['Longitude'], df_stations['Latitude']))# type: ignore 
gdf_stations = gdf_stations.set_crs('EPSG:4326')#type:ignore

#import shapefile of Zwalm: option 2 = best option = EPSG 31370 = BETTER for distances!!
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
gdf_stations_lambert.to_csv(Path("data\Zwalm_data\preprocess_output\zwalm_idw.csv"))


## STEP 2: READING IN THE DATA
pd_rain_data_dict = {}
tmins_list = []
tmaxs_list = []
dateparse_waterinfo = lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.")
for i, filename in enumerate(os.listdir(measurements_file)):
    rain_data = pd.read_csv(measurements_file/filename, sep = ';', skiprows = 8)
    rain_data['#Timestamp'] = rain_data['#Timestamp'].str.rstrip('+01:00') 
    rain_data['#Timestamp'] = rain_data['#Timestamp'].str.rstrip('+02:00')
    rain_data['#Timestamp'] = rain_data['#Timestamp'].apply(dateparse_waterinfo)
    station_name = list_station_names[i]
    #P_name = 'P_' + station_name
    rain_data = rain_data.rename(
        columns = {'#Timestamp':'Time',
        'Value':station_name})

    #replace nans by zeros: will NOT be done for the individual stations!
    #this will only be if ALL stations have no data at that time!
 
    #select data only going from 00 to 23 (analogous to original processing)
    start_date_00, end_date_23 = find_00_23(rain_data['Time'])
    #idea = make hourly timeseries from start -> end. 
    # 1) Only include unique values (so 1 value per hour)
    # 2) If no value present, let it be automatically filled by NaN
    timeseries = pd.date_range(start= start_date_00, end = end_date_23, freq= 'H')
    time_df = pd.DataFrame({'Time':timeseries})
    unique_times, index_unique = np.unique(rain_data['Time'], return_index= True) #select unique timestamps
    rain_data_unique = rain_data.iloc[index_unique,:]
    rain_data_hourly = time_df.merge(rain_data_unique, on = 'Time', how = 'left')
    #again: if this creates NaNs, this is not a problem, since we hope other stations won't have
    #this problem
    tmins_list.append(start_date_00)
    tmaxs_list.append(end_date_23)
    pd_rain_data_dict.update({station_name:rain_data_hourly})

#now make 1 dataframe
timeseries_final = pd.date_range(max(tmins_list),min(tmaxs_list), freq = 'H')
pd_zwalm_hourly_multi = pd.DataFrame({'Time':timeseries_final})
for i, station in enumerate(list_station_names):
     pd_zwalm_hourly_multi = pd_zwalm_hourly_multi.merge(
        pd_rain_data_dict[station][['Time',station]],
        on = 'Time', how = 'left'
        )

# STEP 3: APPLY WEIGHING FACTORS TO DATA
#Idea here: if a Nan value is present for one of the measuring stations,
# then this one is not considered and scaling factors of the others are increased until
# their sum equals one again! 

#naive approach = looping 
rain_matrix = pd_zwalm_hourly_multi[list_station_names].values
pd_IDW_factors = gdf_stations_lambert.filter(regex= '^IDW', axis = 1)
IDW_factors = pd_IDW_factors.values
n_stations, n_exponents_idw = IDW_factors.shape
idw_rain_matrix = np.zeros((rain_matrix.shape[0],n_exponents_idw))
for i in range(rain_matrix.shape[0]):
    row = rain_matrix[i,:]
    bool_nan = np.isnan(row)
    if all(bool_nan):
        idw_rain_matrix[i,:] = 0
    # So only for the timestamps where NO data is present for ALL stations,
    # we will assign 0 
    else:
        correction = np.sum(IDW_factors[~bool_nan.flatten(),:], axis = 0)
        corrected_factors = IDW_factors/correction
        #idea of the above: rescale the non-Nan factors so that their sum 
        #of weighing factors = 1
        row[bool_nan] = 0 #do not take into account the Nan values!
        idw_rain_matrix[i,:] = row@corrected_factors #marix multiplication

pd_idw_rain = pd.DataFrame(idw_rain_matrix, columns = pd_IDW_factors.columns)
pd_zwalm_hourly_final = pd_zwalm_hourly_multi.join(pd_idw_rain)

# STEP 4: ADD FLOW DATA TO THIS FINAL DATAFRAME
