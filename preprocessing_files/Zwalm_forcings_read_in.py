from pywaterinfo import Waterinfo
import pandas as pd
import numpy as np
import datetime
import os
from pathlib import Path 
import pickle 
import pytz
pad = Path(os.getcwd())
if pad.name != "Python":
    pad_correct = Path("../../Python")
    os.chdir(pad_correct)
from functions.pre_processing import make_pd_unique_timesteps

#pyright: reportUnboundVariable=false

#Note: do not run this script too much, or credit will be exceeded
#convention from pywaterinfo issue: always use GMT +1 = UTC +1 for timedata!
#SO ask for data in UTC = 1 hour earlier then we actually want it! 
# => timedelta for shift later
belgian_timezone = pytz.timezone('Europe/Brussels')
#Dates we want to read in
dateparse_waterinfo = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
#Choose your startdates always at 00:00:00 and end_dates at 23:45:00 for 15 min data
t_start = belgian_timezone.localize(dateparse_waterinfo("2012-01-01 00:00:00"))
t_intermed_1 = belgian_timezone.localize(dateparse_waterinfo("2016-12-31 23:45:00"))
t_intermed_2 = belgian_timezone.localize(dateparse_waterinfo("2017-01-01 00:00:00"))
t_end = belgian_timezone.localize(dateparse_waterinfo("2022-11-05 23:45:00"))
t_end_hour = belgian_timezone.localize(dateparse_waterinfo("2022-11-05 23:00:00"))

vmm = Waterinfo('vmm')
hic = Waterinfo('hic')

##############################
# %% Evaporation by Penman-Monteith method
###############################
EP_dict = {}
EP_info_dict = {}
stations = ['Liedekerke','Waregem','Boekhoute']
station_ids = ['ME07_006','ME05_019','ME03_017']
for i in range(len(stations)):
    #Info on the station
    stationsinfo = vmm.get_timeseries_list(station_no = station_ids[i])
    stationsinfo =  stationsinfo[stationsinfo['ts_name'] == 'Monteith.P.15'] 
    #Monteith is supposely the Penmann-Monteith equation! 
    EP_info_dict[stations[i]] = stationsinfo[
        ['station_no','station_name',
        'station_latitude','station_longitude','station_local_x',
        'station_local_y','station_georefsystem','ts_id','ts_unitname']
    ]
    #The timeseries (note values are in mm)
    pd_1 = vmm.get_timeseries_values(
        ts_id = str(int(stationsinfo['ts_id'].values)),#type:ignore
        start = t_start,
        end = t_intermed_1,
        )
    pd_2 = vmm.get_timeseries_values(
        ts_id = str(int(stationsinfo['ts_id'].values)),#type:ignore
        start = t_intermed_2,
        end = t_end
        )
    #merge the 2 pandas dataframes, make unique timesframes
    # + now shift with one hour! GMT -> GMT +1 
    pd_1['Timestamp'] = pd_1['Timestamp'].dt.tz_localize(None)
    pd_2['Timestamp'] = pd_2['Timestamp'].dt.tz_localize(None)
    pd_1['Timestamp'] = pd_1['Timestamp'] + pd.Timedelta(hours = 1)#GMT -> GMT +1 
    pd_2['Timestamp'] = pd_2['Timestamp'] + pd.Timedelta(hours = 1)#GMT -> GMT +1
    pd_1_unique = make_pd_unique_timesteps(pd_1, 'Timestamp',
    t_start.replace(tzinfo=None), t_intermed_1.replace(tzinfo=None), freq = '0.25H')
    pd_2_unique = make_pd_unique_timesteps(pd_2, 'Timestamp',
    t_intermed_2.replace(tzinfo=None), t_end.replace(tzinfo=None), freq = '0.25H')
    pddf = pd.concat([pd_1_unique, pd_2_unique], ignore_index = True)
    pddf_hourly = pddf[['Timestamp','Value']].set_index('Timestamp').resample(
        '1H').agg(pd.DataFrame.sum, skipna=False).reset_index() #sum since in mm, not mm/h!
        #important: if one Nan present when resampling, the entire sum is nan!
    #extra step: set EP lower than 0 to 0
    pddf_hourly.loc[pddf_hourly['Value'] < 0,'Value'] = 0
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
# %% Rainfall
###########
P_dict = {}
P_info_dict = {}
stations = ['Elst','Maarke-Kerkem','Zingem','Ronse']#,'Moerbeke']
providers = ['hic','vmm','vmm','hic']#,'vmm']
station_ids = ['plu06a-1066','P06_014','P06_040','plu12a-1066']#,'P07_021']
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
    pddf['Timestamp'] = pddf['Timestamp'] + pd.Timedelta(hours=1)#GMT-> GMT+1
    pd_unique = make_pd_unique_timesteps(pddf, 'Timestamp',
    t_start.replace(tzinfo=None), t_end_hour.replace(tzinfo=None), '1H')
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

##########
# %% Flow
##########
#no filtering on nans is applied, only making sure that is spans the entire timerange
station = 'Zwalm'
station_id = 'L06_342'
stationsinfo_OG = vmm.get_timeseries_list(station_no = station_id)
ts_name_list = ['P.60','DagGem']
freq_list = ['1H','1D']
output_name_list = ['Q_hour','Q_day']
for i, ts_name in enumerate(ts_name_list):
    stationsinfo =  stationsinfo_OG[stationsinfo_OG['ts_name'] == ts_name]
    stationsinfo =  stationsinfo[stationsinfo['ts_unitsymbol'] == 'm³/s']
    flow_info =  stationsinfo[
            ['station_no','station_name',
            'station_latitude','station_longitude','station_local_x',
            'station_local_y','station_georefsystem','ts_id','ts_unitname']
        ]
    flowdf = vmm.get_timeseries_values(
                ts_id = str(int(stationsinfo['ts_id'].values)),#type:ignore
                start = t_start,
                end = t_end_hour
            )
    flowdf['Timestamp'] = flowdf['Timestamp'].dt.tz_localize(None)
    flowdf['Timestamp'] = flowdf['Timestamp'] + pd.Timedelta(hours = 1)#GMT -> GMT+1
    if ts_name == ts_name_list[0]:
        pd_unique = make_pd_unique_timesteps(flowdf, 'Timestamp',
        t_start.replace(tzinfo=None), t_end_hour.replace(tzinfo=None), freq_list[i])
        if len(pd_unique)%24 != 0:
            raise Warning("""Length of the hourly Dataframe is not dividable
            by 24, check for problems with duplicates""")
    elif ts_name == ts_name_list[1]:
        pd_unique = make_pd_unique_timesteps(flowdf, 'Timestamp',
        t_start.replace(tzinfo=None),t_end_hour.replace(tzinfo = None,hour = 00), freq_list[i])
    pickle_info = output_name_list[i] + '_info.pkl'
    csv_info = output_name_list[i] + '_info.csv'
    pickle_values = output_name_list[i] + '.pkl'
    csv_values = output_name_list[i] + '.csv'
    flow_info.to_pickle(output_folder/pickle_info)
    flow_info.to_csv(output_folder/csv_info, index = False)
    pd_unique.to_pickle(output_folder/pickle_values)
    pd_unique.to_csv(output_folder/csv_values, index = False)

