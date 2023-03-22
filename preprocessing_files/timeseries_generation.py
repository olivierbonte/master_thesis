# %% Read in
import xarray as xr
import pandas as pd
import hvplot.pandas
import hvplot.xarray
import scipy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
pad = Path(os.getcwd())
if pad.name == "preprocessing_files":
    pad_correct = pad.parent
    os.chdir(pad_correct)
write = True

#use decode_coords = 'all' for max compatiblity with rioxarray
s1_xr = xr.open_dataset('data/s0_OpenEO/S0_zwalm_landuse.nc',decode_coords='all')
s1_xr = s1_xr.chunk({'t':20}) #chunck per 20 timesteps
s1_xr_gamma0 = xr.open_dataset('data/g0_OpenEO/g0_zwalm_landuse.nc',decode_coords='all')
s1_xr_gamma0 = s1_xr_gamma0.chunk({'t':20}) #chunck per 20 timesteps
s1_xr['landuse'] = s1_xr['landuse'].astype(np.uint8)
LAI_xr = xr.open_dataset('data/LAI/LAI_cube_Zwalm_landuse.nc', decode_coords='all')
LAI_xr = LAI_xr.chunk('auto')
LAI_xr['landuse'] = LAI_xr['landuse'].astype(np.uint8)

#####################################################
# %% Take averages per landuse category for Sentinel-1
#######################################################
landuseclasses = ['Urban','Forest','Pasture','Agriculture','Water']
landusenumbers = [1,2,3,4,5]
name_list = []
for i in range(len(landusenumbers)):
    VV_abs_temp = s1_xr['VV'].where(s1_xr['landuse'] == landusenumbers[i])
    VH_abs_temp = s1_xr['VH'].where(s1_xr['landuse'] == landusenumbers[i])
    name = 'VV' + landuseclasses[i]
    name_VH = 'VH' + landuseclasses[i] 
    name_list.append(name)
    name_list.append(name_VH)
    s1_xr[name] = 10*np.log10(VV_abs_temp.mean(dim = ['y','x'], skipna=True)) 
    s1_xr[name_VH] = 10*np.log10(VH_abs_temp.mean(dim = ['y','x'], skipna=True)) 
    #also for gamma0
    VV_abs_temp_gamma0 = s1_xr_gamma0['VV'].where(s1_xr_gamma0['landuse'] == landusenumbers[i])
    VH_abs_temp_gamma0 = s1_xr_gamma0['VH'].where(s1_xr_gamma0['landuse'] == landusenumbers[i])
    s1_xr_gamma0[name] = 10*np.log10(VV_abs_temp_gamma0.mean(dim = ['y','x'], skipna=True)) 
    s1_xr_gamma0[name_VH] = 10*np.log10(VH_abs_temp_gamma0.mean(dim = ['y','x'], skipna=True)) 
name_list.append('Orbitdirection')
#general average
s1_xr['VV_avg'] = 10*np.log10(s1_xr['VV'].mean(dim = ['y','x'], skipna=True))
s1_xr['VH_avg'] = 10*np.log10(s1_xr['VH'].mean(dim = ['y','x'], skipna=True))
s1_xr_gamma0['VV_avg'] = 10*np.log10(s1_xr_gamma0['VV'].mean(dim = ['y','x'], skipna=True))
s1_xr_gamma0['VH_avg'] = 10*np.log10(s1_xr_gamma0['VH'].mean(dim = ['y','x'], skipna=True))
s1_xr_gamma0['lia_avg'] = s1_xr_gamma0['local_incidence_angle'].mean(dim = ['y','x'], skipna= True)
#Average over agriculture and pasture combined
s1_xr['VV_past_agr'] = 10*np.log10(s1_xr['VV'].where(
        (s1_xr['landuse'] == 3) | (s1_xr['landuse'] == 4)
    ).mean(
        dim = ['y','x'], skipna=True
    ))
s1_xr['VH_past_agr'] = 10*np.log10(s1_xr['VH'].where(
        (s1_xr['landuse'] == 3) | (s1_xr['landuse'] == 4)
    ).mean(
        dim = ['y','x'], skipna=True
    ))
s1_xr_gamma0['VV_past_agr'] = 10*np.log10(s1_xr_gamma0['VV'].where(
        (s1_xr_gamma0['landuse'] == 3) | (s1_xr_gamma0['landuse'] == 4)
    ).mean(
        dim = ['y','x'], skipna=True
    ))
s1_xr_gamma0['VH_past_agr'] = 10*np.log10(s1_xr_gamma0['VH'].where(
        (s1_xr_gamma0['landuse'] == 3) | (s1_xr_gamma0['landuse'] == 4)
    ).mean(
        dim = ['y','x'], skipna=True
    ))
name_list.append('VV_avg')
name_list.append('VH_avg')
name_list.append('VV_past_agr')
name_list.append('VH_past_agr')
name_list_gamma0 = name_list.copy()
name_list_gamma0.append('lia_avg')

# %% convert to pandas
pd_s1_tseries = s1_xr[name_list].to_pandas()
pd_s1_gamma0_tseries = s1_xr_gamma0[name_list_gamma0].to_pandas()
pd_s1_tseries[name_list].groupby('Orbitdirection').plot()
pd_s1_gamma0_tseries[name_list_gamma0].groupby('Orbitdirection').plot()
timestamps_s1 = pd_s1_tseries.index

###############################################
# %% Take averages per landuse category for LAI
###############################################
frac_required = 0.4
#minimally 80% (changed to 40) of the cells of that landcategory should be non-nan before taking
#into_account as valid timestamp
da_landuse = LAI_xr['landuse']
nr_timesteps_LAI = LAI_xr['LAI'].shape[0]
nr_nonan_list = []
total_pixels = da_landuse.shape[0]*da_landuse.shape[1]
#find timestep with max number of no nan values
for i in range(nr_timesteps_LAI):
    nonan_nr = total_pixels - np.sum(np.isnan(LAI_xr['LAI'].isel(t=i))).values
    nr_nonan_list.append(nonan_nr)
#select first timestep with max amount of cells to create a mask
max_positions = np.where(nr_nonan_list == max(nr_nonan_list))
mask = ~np.isnan(LAI_xr['LAI'].isel(t=max_positions[0][0]))
nr_pixels_landuse_masked = np.sum(mask).values#type:ignore
bool_name_list = []
for i in range(len(landusenumbers)+1): #add 1 iteration for agriculture + pasture
    if i < len(landusenumbers):
        da_landcat = da_landuse.where(da_landuse == landusenumbers[i])
    else:
        da_landcat = da_landuse.where((da_landuse == 3) | (da_landuse == 4))
    da_landcat = da_landcat.where(mask == 1) #only WITHIN catchment!
    if i < len(landusenumbers):
        da_LAI_landcat = LAI_xr['LAI'].where(da_landuse == landusenumbers[i])
    else:
        da_LAI_landcat = LAI_xr['LAI'].where((da_landuse == 3) | (da_landuse == 4))
    da_LAI_landcat = da_LAI_landcat.where(mask == 1) #only WITHIN catchment!
    nr_pixels_landcat = np.sum(~np.isnan(da_landcat)).values#type:ignore
    bool_full_image = []
    for j in range(nr_timesteps_LAI):
        da_LAI_landcat_tj = da_LAI_landcat.isel(t=j)
        nonan_count = np.sum(~np.isnan(da_LAI_landcat_tj)).values#type:ignore
        if nr_pixels_landcat > 0:
            frac = nonan_count/nr_pixels_landcat
            if frac < frac_required:
                bool_full_image.append(0)
            else:
                bool_full_image.append(1)
        else:
            bool_full_image.append(0)
    if i < len(landusenumbers):
        name = 'bool_full_image' + landuseclasses[i]
    else:
        name = 'bool_full_image_past_agr'
    bool_name_list.append(name)
    da_add = xr.DataArray(
        data = bool_full_image,
        dims = ['t'],
        coords = dict(t = LAI_xr['t'].values)
    )#type:ignore
    LAI_xr[name] = da_add

#  Now calculate spatial average per category
name_list_LAI = []
for i in range(len(landusenumbers)+1):
    if i < len(landusenumbers):
        LAI_temp = LAI_xr['LAI_pv'].where(LAI_xr['landuse'] == landusenumbers[i])
    else:
        LAI_temp = LAI_xr['LAI_pv'].where((LAI_xr['landuse'] == 3) | (LAI_xr['landuse'] == 4))
    LAI_temp = LAI_temp.where(LAI_xr[bool_name_list[i]] == 1) #selection based on frac nan!
    if i < len(landusenumbers):
        name = 'LAI' + landuseclasses[i]
    else:
        name = 'LAI_past_agr'
    name_list_LAI.append(name)
    LAI_xr[name] = LAI_temp.mean(dim = ['y','x'],skipna=True)
pd_LAI_tseries = LAI_xr[name_list_LAI].to_pandas()
pd_LAI_tseries[name_list_LAI].plot()

###################################
# %% Interpolating LAI data
# Hans 20/12/2022: calculate average LAI by monthly windows
# if 't' not in set(pd_LAI_tseries.columns):
#     pd_LAI_tseries = pd_LAI_tseries.reset_index()
# pd_LAI_tseries['month'] = pd_LAI_tseries['t'].dt.month.values
# pd_montly_ave = pd_LAI_tseries.groupby('month').mean()
# LAI_columns = pd_LAI_tseries.columns[0:6]
# def apply_monthly_ave(row):
#     nan_bool = row[LAI_columns].isna()
#     nan_colmuns = LAI_columns[nan_bool]
#     month = row['t'].month
#     monthly_average = pd_montly_ave.loc[month]
#     row[nan_colmuns] = monthly_average[nan_colmuns]
#     return row
# pd_LAI_tseries_filled = pd_LAI_tseries[LAI_columns].apply(lambda row: apply_monthly_ave(row), axis = 1)
# pd_LAI_tseries_filled = pd_LAI_tseries_filled.set_index('t')
# pd_LAI_tseries = pd_LAI_tseries.set_index('t')
# pd_LAI_tseries_filled[name_list_LAI].plot()
# Above leads to worse performance => droppend on 21/12/2022

#Neural ode julia uses Steffen's method
#https://gist.github.com/niclasmattsson/7bceb05fba6c71c78d507adae3d29417
# https://en.wikipedia.org/wiki/Monotone_cubic_interpolation 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html 
#test_interp = pd_LAI_tseries.interpolate(mehtod = 'cubicspline')
#monoton_cubic_interp = pd_LAI_tseries.interpolate(method = 'pchip')
#TESTING INTERPOLATION
# no_nan_pasture = pd_LAI_tseries['LAIPasture'].dropna()
# f = scipy.interpolate.PchipInterpolator(no_nan_pasture.index, no_nan_pasture)
# LAI_s1_times = f(timestamps_s1)
# Interpolating on the unknown timestemps!

pd_LAI_tseries_filled = pd_LAI_tseries.copy()
#adding interpolated values to timeseries of Sentinel 1
t_plotting = pd.date_range(pd_LAI_tseries_filled.index[0], timestamps_s1[-1], freq = 'H')#type:ignore
pd_plotting_dict = {}
pd_plotting_dict['t'] = t_plotting
timestamps_s1_gamma0 = pd_s1_gamma0_tseries.index
name_list_LAI_interp = []
for i in range(len(name_list_LAI)):
    LAI_ts = pd_LAI_tseries[name_list_LAI[i]]
    LAI_ts = LAI_ts.dropna() #only fit interpolator on not nans
    if len(LAI_ts) > 0:
        f = scipy.interpolate.PchipInterpolator(LAI_ts.index, LAI_ts)
        LAI_ts_s1 =f(timestamps_s1)
        LAI_ts_s1_gamma0 =f(timestamps_s1_gamma0)
        LAI_filled = f(pd_LAI_tseries_filled.index)
        LAI_plotting = f(t_plotting)
        pd_s1_tseries[name_list_LAI[i]] = LAI_ts_s1
        pd_s1_gamma0_tseries[name_list_LAI[i]] = LAI_ts_s1_gamma0
        pd_LAI_tseries_filled[name_list_LAI[i]] = LAI_filled
        pd_plotting_dict[name_list_LAI[i]] = LAI_plotting
        name_list_LAI_interp.append(name_list_LAI[i])

#plot the filled datastet
fig, ax = plt.subplots(figsize = (10,7))
pd_LAI_tseries[name_list_LAI].plot(ax = ax, title = 'Original data')
ax.set_ylabel('LAI')
fig, ax = plt.subplots(figsize = (10,7))
pd_LAI_tseries_filled[name_list_LAI].plot(ax = ax,title = 'Spline interpolation')
ax.set_ylabel('LAI')

## plot interpolated values on original data
fig, ax = plt.subplots(figsize = (15,10))
pd_LAI_tseries[name_list_LAI].plot(ax = ax, marker = '.', linestyle = 'None')
pd_s1_tseries[name_list_LAI_interp].plot.line(ax = ax, marker = '*', linestyle = 'None')

## Final plot
pd_plotting = pd.DataFrame(pd_plotting_dict)
pd_plotting = pd_plotting.set_index('t')
fig, ax = plt.subplots(figsize = (10,7))
pd_LAI_tseries[name_list_LAI_interp].plot(ax = ax, marker = '.', linestyle = 'None')
colors_used = [plt.gca().lines[i].get_color() for i in range(len(landusenumbers)-1)]
pd_plotting[name_list_LAI_interp].plot(ax = ax, color = colors_used)
og_names = ['Urban','Forest','Pasture','Agriculture','Pasture and Agriculture']
interpol_names = ['Urban: interpolated','Forest: interpolated','Pasture: interpolated',
'Agriculture: interpolated','Pasture and agriuclutre: interpolated']
ax.legend(og_names + interpol_names, ncol = 2)
#Even further optimised this figure in Chapter_data.ipynb 

# # %% COMPARING BELOW WITH HVPLOT
# pd_LAI_tseries[name_list_LAI].hvplot(frame_width = 800, title = 'original')
# # %%
# pd_LAI_tseries_filled[name_list_LAI].hvplot(frame_width = 800, title = 'Spline interpolation')

# %% write out timeseries
if write: 
    pd_s1_tseries.to_csv('data/s0_OpenEO/s1_timeseries.csv')
    pd_s1_gamma0_tseries.to_csv('data/g0_OpenEO/s1_g0_timeseries.csv')
    pd_LAI_tseries.to_csv('data/LAI/LAI_timeseries.csv')
    pd_plotting.to_csv('data/LAI/LAI_plotting.csv')
    print('Dataframes saved to csv')
    pd_LAI_tseries.to_pickle('data/LAI/LAI_timeseries.pkl')
    pd_plotting.to_pickle('data/LAI/LAI_plotting.pkl')
# %%
