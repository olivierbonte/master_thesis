# %% Load in modules
import pandas as pd
import geopandas as gpd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
pad = Path(os.getcwd())
if pad.name == "ml_observation_operator":
    pad_correct = pad.parent
    os.chdir(pad_correct)
from functions.PDM import PDM

# %% PDM
#Forcing data
preprocess_output_folder = Path('data/Zwalm_data/preprocess_output')
p_zwalm = pd.read_pickle(preprocess_output_folder/'zwalm_p_thiessen.pkl')
ep_zwalm = pd.read_pickle(preprocess_output_folder/'zwalm_ep_thiessen.pkl')
# print(f'{p_zwalm=}')
# print('\n')
# print(f'{ep_zwalm=}')
#Parameterset
param = pd.read_csv("data/Zwalm_PDM_parameters/p1_opt_param_mNSE_PSO_70_particles_qconst_strict.csv", index_col = False)
param = param.drop(param.columns[0], axis = 1)
#print(f'{param=}')

#Area Zwalm
zwalm_shape = gpd.read_file('data/Zwalm_shape/zwalm_shapefile_emma_31370.shp')
area_zwalm_new = np.single(zwalm_shape.area[0]/10**6)

# %% Features for data-driven model
#use the gamma0 data for this! 
features = pd.read_csv('data/g0_OpenEO/s1_g0_timeseries.csv', parse_dates=True)
features = features.set_index('t')
features.index = pd.to_datetime(features.index)
#print(features.columns)
cols = features.columns[features.columns.str.endswith(
    ('Pasture','Agriculture','Forest','past_agr')
    )]
#Make dummy variables of the orbit direction
orb_dir_dummies = pd.get_dummies(features.Orbitdirection)
orb_dir_dummies = orb_dir_dummies.set_index(features.index)
features = pd.concat([features[cols],orb_dir_dummies],axis = 1)
#print(f'{features=}')

# %% Run PDM
deltat = np.array(1,dtype = np.float32) #hour
deltat_out = np.array(24, dtype = np.float32) #daily averaging
pd_zwalm_out_day = PDM(P = p_zwalm['P_thiessen'].values, 
    EP = ep_zwalm['EP_thiessen'].values,
    t = p_zwalm['Timestamp'].values,
    area = area_zwalm_new, deltat = deltat, deltatout = deltat_out ,
    parameters = param, m = 3)
pd_zwalm_out_day = pd_zwalm_out_day.set_index('Time')
pd_zwalm_out_day['Cstar'].plot(title = 'C*', ylabel = '[mm]')
pd_zwalm_out_day.head()
#print(f'{pd_zwalm_out_day=}')

Cstar = pd_zwalm_out_day['Cstar']
#print(f'{Cstar=}')

# %% Add time related features
# deltat_t
deltat_t_arr = np.zeros((features.shape[0],))
for i in range(features.shape[0]):
    if i == 0:
        dt_temp = 0
    else:
        dt_temp = features.index[i] - features.index[i-1]#type:ignore
        dt_temp = dt_temp.total_seconds()/(3600*24)
    deltat_t_arr[i] = dt_temp
features['delta_t'] = deltat_t_arr

#sine and cosine features
features['year_sin'] = np.sin(features.index.day_of_year*2*np.pi/365.2425)#type:ignore
features['year_cos'] = np.cos(features.index.day_of_year*2*np.pi/365.2425)#type:ignore


# %% Split up dataset in trainig and validation part
t1_features = features.index[0]
print('day 1 SAR data: ' + str(t1_features))
tend_features = features.index[-1]
print('last day SAR data: ' + str(tend_features))
nr_days = tend_features - t1_features#type:ignore
nr_years = nr_days.total_seconds()/(3600*24*365.25)
print('years of SAR data: ' + str(nr_years))

#5.5 years of trainig data, remaining 2 as validation data
tend_calibration = pd.Timestamp(datetime(year = 2020, month = 12, day = 31))
tbegin_validation = pd.Timestamp(datetime(year = 2021, month = 1, day = 1))

print('Calibration period: ' + str(t1_features) + ' until ' + str(tend_calibration))
print('Validation period: ' + str(tbegin_validation) + ' until ' + str(tend_features))

X_train = features[t1_features:tend_calibration]
X_test = features[tbegin_validation:tend_features]
X_full = features[t1_features:tend_features]
 
#select only on days with available training data! 
y_train = Cstar[X_train.index]
y_test = Cstar[X_test.index]
y_full = Cstar[X_full.index]

print('----------------------------------------------------------\n')
print('Data has been split up in X_train, X_test, y_train and y_test')
print('------------------------------------------------------------\n')


# %% Save out features 
pad = Path(os.getcwd())
ML_data_pad = Path("data/Zwalm_data/ML_data")
if not os.path.exists(ML_data_pad):
    os.mkdir(ML_data_pad)

X_train.to_pickle(ML_data_pad/'X_train.pkl')
X_test.to_pickle(ML_data_pad/'X_test.pkl')
X_full.to_pickle(ML_data_pad/"X_full.pkl")

y_train.to_pickle(ML_data_pad/"y_train.pkl")
y_test.to_pickle(ML_data_pad/'y_test.pkl')
y_full.to_pickle(ML_data_pad/"y_full.pkl")

Cstar.to_pickle(ML_data_pad/"Cstar.pkl")