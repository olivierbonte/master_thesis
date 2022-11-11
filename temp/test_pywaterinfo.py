from pywaterinfo import Waterinfo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def make_pd_unique_timesteps(pddf, t_column_name, t_start, t_end, freq):
    """Transform pandas Dateframe so that you left join it on a
    desired timeseries going from t_start to t_end at desired frequency

    Parameters
    ------------
    pddf: pandas.DataFrame

    t_colum_name: string
        name of where time (as datetime) is located in pd
    t_start: datetime
    t_end: datetime
    freq: string
        desired frequency of timeseries
    Returns
    --------
    pddf_unique: pandas.DataFrame  
        unique pandas Dataframe
    """
    timeseries = pd.DataFrame({t_column_name:pd.date_range(
    t_start, t_end, freq = freq)})
    pddf_unique = timeseries.merge(
        pddf, on = t_column_name, how = 'left'
    )
    return pddf_unique



vmm = Waterinfo("vmm")
stationsinfo = vmm.get_timeseries_list(station_no= "ME07_006") #Liedekerke
print(stationsinfo.columns)
print(stationsinfo['parametertype_name'].unique())
# bool_PET = stationsinfo['parametertype_name'] == 'PET'
# stationsinfo_PET = stationsinfo[bool_PET]
# print(stationsinfo_PET['ts_spacing'].unique())
# all_ids = stationsinfo["ts_id"]
# bool_15M = stationsinfo_PET["ts_spacing"] == 'PT15M'
# stationsinfo_PET_15M = stationsinfo_PET[bool_15M]
# bool_penman = 'Penman.P.15' == stationsinfo_PET_15M['ts_name']
# stationsinfo_PET_15M_Penman = stationsinfo_PET_15M[bool_penman]
# df_15m = vmm.get_timeseries_values(
#     ts_id = 204341,
#     start = "2012-01-01",
#     end = "2022-11-06")

#new trial
dateparse_waterinfo = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
#Choose your startdates always at 00:00:00 and end_dates at 23:45:00 for 15 min data!
t_start = dateparse_waterinfo("2012-01-01 00:00:00")
t_intermed_1 = dateparse_waterinfo("2016-12-31 23:45:00")
t_intermed_2 = dateparse_waterinfo("2017-01-01 00:00:00")
t_end = dateparse_waterinfo("2022-11-05 23:45:00")

#do not run this script too much, or credit will be exceeded
stationsinfo_PET_15M_Penman_Liedekerke = stationsinfo[stationsinfo['ts_name'] == 'Penman.P.15']
pd_liedekerke_1 = vmm.get_timeseries_values(
    ts_id = str(int(stationsinfo_PET_15M_Penman_Liedekerke['ts_id'].values)),#type:ignore
    start = t_start,
    end = t_intermed_1
    )
pd_liedekerke_2 = vmm.get_timeseries_values(
    ts_id = str(int(stationsinfo_PET_15M_Penman_Liedekerke['ts_id'].values)),#type:ignore
    start = t_intermed_2,
    end = t_end
    )

#merge the 2 pandas dataframes, make unique timesframes
timeseries_1 = pd.DataFrame({'Timestamp':pd.date_range(
    t_start, t_intermed_1, freq = '0.25H', 
)}) 
pd_liedekerke_1['Timestamp'] = pd_liedekerke_1['Timestamp'].dt.tz_localize(None) #drop timezone info
pd_liedekerke_1_unique = timeseries_1.merge(
    pd_liedekerke_1,
    on = 'Timestamp', how = 'left'
)
pd_liedekerke_2['Timestamp'] = pd_liedekerke_2['Timestamp'].dt.tz_localize(None)
pd_liedekerke_2_unique = make_pd_unique_timesteps(pd_liedekerke_2, 'Timestamp',
t_intermed_2, t_end, freq = '0.25H')
pd_liedekere = pd.concat([pd_liedekerke_1_unique, pd_liedekerke_2_unique], ignore_index = True)

def custom_nanmean(arraylike):
    return np.nanmean(arraylike)
pd_liedekere_hourly = pd_liedekere[['Timestamp','Value']].set_index('Timestamp').resample(
    '1H').apply(custom_nanmean).reset_index()
pd_liedekere_hourly.to_pickle('data/pickle_test.pkl')