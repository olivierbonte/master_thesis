import pandas as pd
from pathlib import Path
import datetime
import numpy as np

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
 

