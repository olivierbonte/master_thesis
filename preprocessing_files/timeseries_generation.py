# %% Read in
import xarray as xr
import hvplot.pandas
import hvplot.xarray
import scipy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
pad = Path(os.getcwd())
if pad.name != "Python":
    pad_correct = Path("../../Python")
    os.chdir(pad_correct)

#use decode_coords = 'all' for max compatiblity with rioxarray
s1_xr = xr.open_dataset('data/s0_OpenEO/S0_zwalm_landuse.nc',decode_coords='all')
s1_xr = s1_xr.chunk({'t':20}) #chunck per 20 timesteps
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
    name = 'VV' + landuseclasses[i]
    name_list.append(name)
    s1_xr[name] = 10*np.log10(VV_abs_temp.mean(dim = ['y','x'], skipna=True)) 
name_list.append('Orbitdirection')
#general average
s1_xr['VV_avg'] = 10*np.log10(s1_xr['VV'].mean(dim = ['y','x'], skipna=True))
name_list.append('VV_avg')
pd_s1_tseries = s1_xr[name_list].to_pandas()
pd_s1_tseries[name_list].groupby('Orbitdirection').plot()
timestamps_s1 = pd_s1_tseries.index

###############################################
# %% Take averages per landuse category for LAI
###############################################
frac_required = 0.8 
#minimally 80% of the cells of that landcategory should be non-nan before taking
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
nr_pixels_landuse_masked = np.sum(mask).values
bool_name_list = []
for i in range(len(landusenumbers)):
    da_landcat = da_landuse.where(da_landuse == landusenumbers[i])
    da_landcat = da_landcat.where(mask == 1) #only WITHIN catchment!
    da_LAI_landcat = LAI_xr['LAI'].where(da_landuse == landusenumbers[i])
    da_LAI_landcat = da_LAI_landcat.where(mask == 1) #only WITHIN catchment!
    nr_pixels_landcat = np.sum(~np.isnan(da_landcat)).values
    bool_full_image = []
    for j in range(nr_timesteps_LAI):
        da_LAI_landcat_tj = da_LAI_landcat.isel(t=j)
        nonan_count = np.sum(~np.isnan(da_LAI_landcat_tj)).values
        if nr_pixels_landcat > 0:
            frac = nonan_count/nr_pixels_landcat
            if frac < frac_required:
                bool_full_image.append(0)
            else:
                bool_full_image.append(1)
        else:
            bool_full_image.append(0)
    name = 'bool_full_image' + landuseclasses[i]
    bool_name_list.append(name)
    da_add = xr.DataArray(
        data = bool_full_image,
        dims = ['t'],
        coords = dict(t = LAI_xr['t'].values)
    )
    LAI_xr[name] = da_add

#  Now calculate spatial average per category
name_list_LAI = []
for i in range(len(landusenumbers)):
    LAI_temp = LAI_xr['LAI_pv'].where(LAI_xr['landuse'] == landusenumbers[i])
    LAI_temp = LAI_temp.where(LAI_xr[bool_name_list[i]] == 1) #selection based on frac nan!
    name = 'LAI' + landuseclasses[i]
    name_list_LAI.append(name)
    LAI_xr[name] = LAI_temp.mean(dim = ['y','x'],skipna=True)
pd_LAI_tseries = LAI_xr[name_list_LAI].to_pandas()
pd_LAI_tseries[name_list_LAI].plot()

###################################
# %% Interpolating LAI data to 

#Neural ode julia uses Steffen's method
#https://gist.github.com/niclasmattsson/7bceb05fba6c71c78d507adae3d29417
# https://en.wikipedia.org/wiki/Monotone_cubic_interpolation 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html 
test_interp = pd_LAI_tseries.interpolate(mehtod = 'cubicspline')
monoton_cubic_interp = pd_LAI_tseries.interpolate(method = 'pchip')


#TESTING INTERPOLATION
no_nan_pasture = pd_LAI_tseries['LAIPasture'].dropna()
f = scipy.interpolate.PchipInterpolator(no_nan_pasture.index, no_nan_pasture)
LAI_s1_times = f(timestamps_s1)

#adding interpolated values to timeseries of Sentinel 1
for i in range(len(landusenumbers)):
    LAI_ts = pd_LAI_tseries[name_list_LAI[i]]
    LAI_ts = LAI_ts.dropna() #only fit interpolator on not nans
    if len(LAI_ts) > 0:
        f = scipy.interpolate.PchipInterpolator(LAI_ts.index, LAI_ts)
        LAI_ts_s1 =f(timestamps_s1)
        pd_s1_tseries[name_list_LAI[i]] = LAI_ts_s1

## plot interpolated values on original data
fig, ax = plt.subplots(figsize = (15,10))
pd_LAI_tseries[name_list_LAI].plot(ax = ax)
pd_s1_tseries[name_list_LAI[:-1]].plot.line(marker = '.', ax = ax, linestyle = 'None')

# %% write out timeseries
pd_s1_tseries.to_csv('data/s0_OpenEO/s1_timeseries.csv')
pd_LAI_tseries.to_csv('data/LAI/LAI_timeseries.csv')
# %%
