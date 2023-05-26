# %% Load in modules
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import zipfile
from pathlib import Path
from datetime import datetime
pad = Path(os.getcwd())
if pad.name == "ml_observation_operator":
    pad_correct = pad.parent
    os.chdir(pad_correct)
from functions.PDM import PDM

# %% PDM
# Forcing data
preprocess_output_folder = Path('data/Zwalm_data/preprocess_output')
p_zwalm = pd.read_pickle(preprocess_output_folder / 'zwalm_p_thiessen.pkl')
ep_zwalm = pd.read_pickle(preprocess_output_folder / 'zwalm_ep_thiessen.pkl')
# print(f'{p_zwalm=}')
# print('\n')
# print(f'{ep_zwalm=}')
# Parameterset
param = pd.read_csv("data/Zwalm_PDM_parameters/NM_opt_param.csv")
# param = param.drop(param.columns[0], axis = 1)
# print(f'{param=}')

# Area Zwalm
zwalm_shape = gpd.read_file('data/Zwalm_shape/zwalm_shapefile_emma_31370.shp')
area_zwalm_new = np.single(zwalm_shape.area[0] / 10**6)

# %% Features for data-driven model
# use the gamma0 data for this!
features = pd.read_csv('data/g0_OpenEO/s1_g0_timeseries.csv', parse_dates=True)
features = features.set_index('t')
features.index = pd.to_datetime(features.index)
# print(features.columns)
cols = features.columns[features.columns.str.endswith(
    ('Pasture', 'Agriculture', 'Forest', 'past_agr')
)]
# Make dummy variables of the orbit direction
orb_dir_dummies = pd.get_dummies(features.Orbitdirection)
orb_dir_dummies = orb_dir_dummies.set_index(features.index)
features = pd.concat([features[cols], orb_dir_dummies], axis=1)
# print(f'{features=}')
# Update 09/05: drop the descending feature, one suffices (same information carried)
features = features.drop('descending', axis=1)

# %% Run PDM
deltat = np.array(1, dtype=np.float32)  # hour
deltat_out = np.array(24, dtype=np.float32)  # daily averaging
pd_zwalm_out_day = PDM(P=p_zwalm['P_thiessen'].values,
                       EP=ep_zwalm['EP_thiessen'].values,
                       t=p_zwalm['Timestamp'].values,
                       area=area_zwalm_new, deltat=deltat, deltatout=deltat_out,
                       parameters=param, m=3)
pd_zwalm_out_day = pd_zwalm_out_day.set_index('Time')
pd_zwalm_out_day['Cstar'].plot(title='C*', ylabel='[mm]')
pd_zwalm_out_day.head()
# print(f'{pd_zwalm_out_day=}')

Cstar = pd_zwalm_out_day['Cstar']
# print(f'{Cstar=}')

# %% Add time related features
# deltat_t
deltat_t_arr = np.zeros((features.shape[0],))
for i in range(features.shape[0]):
    if i == 0:
        dt_temp = 0
    else:
        dt_temp = features.index[i] - features.index[i - 1]  # type:ignore
        dt_temp = dt_temp.total_seconds() / (3600 * 24)
    deltat_t_arr[i] = dt_temp
features['delta_t'] = deltat_t_arr

# sine and cosine features
features['year_sin'] = np.sin(
    features.index.day_of_year * 2 * np.pi / 365.2425)  # type:ignore
features['year_cos'] = np.cos(
    features.index.day_of_year * 2 * np.pi / 365.2425)  # type:ignore


# %% Split up dataset in trainig and validation part
t1_features = features.index[0]
print('day 1 SAR data: ' + str(t1_features))
tend_features = features.index[-1]
print('last day SAR data: ' + str(tend_features))
nr_days = tend_features - t1_features  # type:ignore
nr_years = nr_days.total_seconds() / (3600 * 24 * 365.25)
print('years of SAR data: ' + str(nr_years))

# 5.5 years of trainig data, remaining 2 as validation data
tend_calibration = pd.Timestamp(datetime(year=2020, month=12, day=31))
tbegin_validation = pd.Timestamp(datetime(year=2021, month=1, day=1))

print('Calibration period: ' + str(t1_features) +
      ' until ' + str(tend_calibration))
print('Validation period: ' + str(tbegin_validation) +
      ' until ' + str(tend_features))

X_train_all = features[t1_features:tend_calibration]
X_test_all = features[tbegin_validation:tend_features]
X_full_all = features[t1_features:tend_features]

# select only on days with available training data!
y_train = Cstar[X_train_all.index]
y_test = Cstar[X_test_all.index]
y_full = Cstar[X_full_all.index]

print('----------------------------------------------------------\n')
print('Data has been split up in X_train, X_test, y_train and y_test')
print('------------------------------------------------------------\n')

# %% Add alternative split up for the time switch experiment
# Keep same number of training and test features
n_test = len(y_test)
X_test_all_bis = X_train_all[0:n_test]
X_train_all_bis = pd.concat([X_train_all[n_test:], X_test_all])

y_train_bis = Cstar[X_train_all_bis.index]
y_test_bis = Cstar[X_test_all_bis.index]

print('----------------------------------------------------------\n')
print('Data has been split up in an alternative X_train, X_test, y_train and y_test')
print('------------------------------------------------------------\n')

# %% Differentiate between the full dataset and the smaller, aggregated dataset
X_train = X_train_all.iloc[:, ~X_train_all.columns.str.endswith('past_agr')]
X_test = X_test_all.iloc[:, ~X_test_all.columns.str.endswith('past_agr')]
X_train_bis = X_train_all_bis.iloc[:, ~X_train_all_bis.columns.str.endswith(
    'past_agr')]
X_test_bis = X_test_all_bis.iloc[:, ~X_test_all_bis.columns.str.endswith(
    'past_agr')]
X_full = X_full_all.iloc[:, ~X_full_all.columns.str.endswith('past_agr')]

X_train_small = X_train_all.iloc[:, ~X_train_all.columns.str.endswith(
    ('Forest', 'Pasture', 'Agriculture'))]
X_test_small = X_test_all.iloc[:, ~X_test_all.columns.str.endswith(
    ('Forest', 'Pasture', 'Agriculture'))]
X_train_small_bis = X_train_all_bis.iloc[:, ~X_train_all.columns.str.endswith(
    ('Forest', 'Pasture', 'Agriculture'))]
X_test_small_bis = X_test_all.iloc[:, ~X_test_all_bis.columns.str.endswith(
    ('Forest', 'Pasture', 'Agriculture'))]
X_full_small = X_full_all.iloc[:, ~X_full_all.columns.str.endswith(
    ('Forest', 'Pasture', 'Agriculture'))]


# %% Save out features
pad = Path(os.getcwd())
ML_data_pad = Path("data/Zwalm_data/ML_data")
if not os.path.exists(ML_data_pad):
    os.mkdir(ML_data_pad)

X_train_all.to_pickle(ML_data_pad / 'X_train_all.pkl')
X_test_all.to_pickle(ML_data_pad / 'X_test_all.pkl')
X_train_all_bis.to_pickle(ML_data_pad / 'X_train_all_bis.pkl')
X_test_all_bis.to_pickle(ML_data_pad / 'X_test_all_bis.pkl')
X_full_all.to_pickle(ML_data_pad / "X_full_all.pkl")

X_train.to_pickle(ML_data_pad / 'X_train.pkl')
X_test.to_pickle(ML_data_pad / 'X_test.pkl')
X_train_bis.to_pickle(ML_data_pad / 'X_train_bis.pkl')
X_test_bis.to_pickle(ML_data_pad / 'X_test_bis.pkl')
X_full.to_pickle(ML_data_pad / "X_full.pkl")

X_train_small.to_pickle(ML_data_pad / 'X_train_small.pkl')
X_test_small.to_pickle(ML_data_pad / 'X_test_small.pkl')
X_train_small_bis.to_pickle(ML_data_pad / 'X_train_small_bis.pkl')
X_test_small_bis.to_pickle(ML_data_pad / 'X_test_small_bis.pkl')
X_full_small.to_pickle(ML_data_pad / "X_full_small.pkl")

y_train.to_pickle(ML_data_pad / "y_train.pkl")
y_test.to_pickle(ML_data_pad / 'y_test.pkl')
y_train_bis.to_pickle(ML_data_pad / 'y_train_bis.pkl')
y_test_bis.to_pickle(ML_data_pad / 'y_test_bis.pkl')
y_full.to_pickle(ML_data_pad / "y_full.pkl")

Cstar.to_pickle(ML_data_pad / "Cstar.pkl")

# export pd_zwalm_out_day
pd_zwalm_out_day.to_pickle("data/Zwalm_data/pd_zwalm_out_day.pkl")

# Get data from Zenodo with models/hyperparameters
#only download if the zip folders does not already exist!
zip_folder = Path('data/ml_obs_op_data.zip')
if not os.path.exists(zip_folder):
    os.run("zenodo_get data 10.5281/zenodo.7973569")
    with zipfile.ZipFile("data/ml_obs_op_data.zip", 'r') as zip_ref:
        zip_ref.extractall('data/ml_obs_op_data')
    