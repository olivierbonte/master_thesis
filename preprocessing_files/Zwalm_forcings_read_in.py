from pywaterinfo import Waterinfo
import pandas as pd
import numpy as np
import datetime
import os
from pathlib import Path 
import pickle 

pad = Path(os.getcwd())
if pad.name != "Python":
    pad_correct = Path("../../Python")
    os.chdir(pad_correct)
from functions.pre_processing import make_pd_unique_timesteps

#Note: do not run this script too much, or credit will be exceeded
#Dates we want to read in
dateparse_waterinfo = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
#Choose your startdates always at 00:00:00 and end_dates at 23:45:00 for 15 min data!
t_start = dateparse_waterinfo("2012-01-01 00:00:00")
t_intermed_1 = dateparse_waterinfo("2016-12-31 23:45:00")
t_intermed_2 = dateparse_waterinfo("2017-01-01 00:00:00")
t_end = dateparse_waterinfo("2022-11-05 23:45:00")
t_end_hour = dateparse_waterinfo("2022-11-05 23:00:00")


vmm = Waterinfo('vmm')
hic = Waterinfo('hic')

##############################
# Evaporation by Penman method
###############################
EP_dict = {}
EP_info_dict = {}
stations = ['Liedekerke','Waregem']
station_ids = ['ME07_006','ME05_019']
for i in range(len(stations)):
    #Info on the station
    stationsinfo = vmm.get_timeseries_list(station_no = station_ids[i])
    stationsinfo =  stationsinfo[stationsinfo['ts_name'] == 'Penman.P.15']
    EP_info_dict[stations[i]] = stationsinfo[
        ['station_no','station_name',
        'station_latitude','station_longitude','station_local_x',
        'station_local_y','station_georefsystem','ts_id','ts_unitname']
    ]
    #The timeseries (note values are in mm)
    pd_1 = vmm.get_timeseries_values(
        ts_id = str(int(stationsinfo['ts_id'].values)),#type:ignore
        start = t_start,
        end = t_intermed_1
        )
    pd_2 = vmm.get_timeseries_values(
        ts_id = str(int(stationsinfo['ts_id'].values)),#type:ignore
        start = t_intermed_2,
        end = t_end
        )

    #merge the 2 pandas dataframes, make unique timesframes
    pd_1['Timestamp'] = pd_1['Timestamp'].dt.tz_localize(None)
    pd_2['Timestamp'] = pd_2['Timestamp'].dt.tz_localize(None)
    pd_1_unique = make_pd_unique_timesteps(pd_1, 'Timestamp',
    t_start, t_intermed_1, freq = '0.25H')
    pd_2_unique = make_pd_unique_timesteps(pd_2, 'Timestamp',
    t_intermed_2, t_end, freq = '0.25H')
    pddf = pd.concat([pd_1_unique, pd_2_unique], ignore_index = True)
    pddf_hourly = pddf[['Timestamp','Value']].set_index('Timestamp').resample(
        '1H').apply(np.sum).reset_index() #sum since in mm, not mm/h!
    if len(pddf_hourly)%24 != 0:
        raise Warning("""Length of the hourly Dataframe is not dividable
        by 24, check for problems with duplicates""")
    EP_dict[stations[i]] = pddf_hourly

output_folder = Path('data/Zwalm_data/pywaterinfo_output')
with open(output_folder/'EP_dict.pickle', 'wb') as handle:
    pickle.dump(EP_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(output_folder/'EP_info_dict.pickle', 'wb') as handle:
    pickle.dump(EP_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

###########
# Rainfall
###########
P_dict = {}
P_info_dict = {}
stations = ['Elst','Maarke-Kerkem','Zingem','Ronse']
providers = ['hic','vmm','vmm','hic']
station_ids = ['plu06a-1066','P06_014','P06_040','plu12a-1066']
for i in range(len(stations)):
    # Info on the station
    if providers[i] == 'vmm':
        stationsinfo = vmm.get_timeseries_list(station_no = station_ids[i])
        stationsinfo = stationsinfo[stationsinfo['ts_name'] == 'P.60']
        stationsinfo = stationsinfo[stationsinfo['ts_unitsymbol'] == 'mm']
    elif providers[i] == 'hic':
        stationsinfo = hic.get_timeseries_list(station_no = station_ids[i])
        stationsinfo = stationsinfo[stationsinfo['ts_name'] == '60Tot']
    else:
        raise ValueError('No appropriate data provider is given')
    P_info_dict[stations[i]] = stationsinfo[
        ['station_no','station_name',
        'station_latitude','station_longitude','station_local_x',
        'station_local_y','station_georefsystem','ts_id','ts_unitname']
    ]

    #The timeseries (note values are in mm, equivalent to mm/h for 1h resolution)
    if providers[i] == 'vmm':
        pddf = vmm.get_timeseries_values(
            ts_id = str(int(stationsinfo['ts_id'].values)),#type:ignore
            start = t_start,
            end = t_end_hour
        )
    elif providers[i] == 'hic':
        pddf = hic.get_timeseries_values(
            ts_id = str(int(stationsinfo['ts_id'].values)),#type:ignore
            start = t_start,
            end = t_end_hour
        )
    else:
        raise ValueError('No appropriate data provider is given')
    pddf['Timestamp'] = pddf['Timestamp'].dt.tz_localize(None)
    pd_unique = make_pd_unique_timesteps(pddf, 'Timestamp',
    t_start, t_end_hour, '1H')
    if len(pd_unique)%24 != 0:
        raise Warning("""Length of the hourly Dataframe is not dividable
        by 24, check for problems with duplicates""")
    P_dict[stations[i]] = pd_unique
    #check on length
    if i >= 1:
        if len(P_dict[stations[i]]) != len(P_dict[stations[i-1]]):
            raise Warning("Timeseries of " + stations[i] + " and " +
            stations[i-1] + " are not the same!")

with open(output_folder/'P_dict.pickle', 'wb') as handle:
    pickle.dump(P_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(output_folder/'P_info_dict.pickle', 'wb') as handle:
    pickle.dump(P_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



