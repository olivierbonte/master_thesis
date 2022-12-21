# %% [markdown]
# # Tesiting PSO as a calibration function

# %%
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
import hvplot
import hvplot.pandas
import scipy
import winsound 
import warnings
import pyswarms as ps
pad = Path(os.getcwd())
if pad.name != "Python":
    pad_correct = Path("../../Python")
    os.chdir(pad_correct)
from functions.PDM import PDM, PDM_calibration_wrapper_PSO
from functions.performance_metrics import NSE, mNSE


warnings.filterwarnings(action = 'ignore', category= RuntimeWarning)
warnings.filterwarnings(action = 'ignore', category= UserWarning)
parameters_initial = pd.DataFrame({
    'cmax': 400.60999,
    'cmin':87.67600,
    'b':0.60000,
    'be':3.00000,
    'k1':8.00000,
    'k2':0.70000,
    'kb':5.04660,
    'kg':9000.00000,
    'St': 0.43043,
    'bg':1.00000,
    'tdly':2.00000,
    'qconst':0.00000,
    #'rainfac':0.00000 THIS IS NOT USED!
}, dtype = np.float32, index =[0])
print(parameters_initial)

area_zwalm_initial = np.single(109.2300034)
zwalm_shape = gpd.read_file('data/Zwalm_shape/zwalm_shapefile_emma_31370.shp')
area_zwalm_new = np.single(zwalm_shape.area[0]/10**6)
print('Area of the Zwalm by Cabus: ' + str(area_zwalm_initial) + '[km^2]')
print('Area of the Zwalm by shapefile: ' + str(area_zwalm_new) + '[km^2]')

# %%
preprocess_output_folder = Path('data/Zwalm_data/preprocess_output')
p_zwalm = pd.read_pickle(preprocess_output_folder/'zwalm_p_thiessen.pkl')
ep_zwalm = pd.read_pickle(preprocess_output_folder/'zwalm_ep_thiessen.pkl')
ep_zwalm.loc[ep_zwalm['EP_thiessen'] <0, 'EP_thiessen'] = 0 #ADVICE OF NIKO 21/12/2022
#Temporary fix! 
#ep_zwalm.loc[np.isnan(ep_zwalm['EP_thiessen']),'EP_thiessen'] = 0

pywaterinfo_output_folder = Path("data/Zwalm_data/pywaterinfo_output")
Q_day = pd.read_pickle(pywaterinfo_output_folder/"Q_day.pkl")
Q_day = Q_day.set_index('Timestamp')
warmup_months = 9
start_p1 = p_zwalm['Timestamp'].iloc[0]
start_endofwarmup_p1 = start_p1 + relativedelta(months = warmup_months)
end_p1 =  pd.Timestamp(datetime(year = 2017, month = 12, day = 31, hour = 23))
print('Characteristics of period 1: start = '  + str(start_p1) + ', start of post warmup = ' + str(start_endofwarmup_p1) + ' and end = ' + str(end_p1))

start_p2 = pd.Timestamp(datetime(year = 2018, month= 1, day = 1, hour = 0))
start_endofwarmup_p2 = start_p2 + relativedelta(months = warmup_months)
end_p2 = p_zwalm['Timestamp'].iloc[-1]
print('Characteristics of period 2: start = '  + str(start_p2) + ', start of post warmup = ' + str(start_endofwarmup_p2) + ' and end = ' + str(end_p2))

p1_period_excl_warmup = pd.date_range(start_endofwarmup_p1,end_p1,
freq = 'D') #used for scoring the model 
p1_period = pd.date_range(start_p1, end_p1, freq = 'H')
p2_period_excl_warmup = pd.date_range(start_endofwarmup_p2,end_p2,
freq = 'D') #used for scoring the model 
p2_period = pd.date_range(start_p2, end_p2, freq = 'H')
p_all_nowarmup = pd.date_range(start_endofwarmup_p1, end_p2)
p_all = pd.date_range(start_p1, end_p2)

#now subdivide ep data on p1 and p2
#for ease of selecting data, set time as index!
#select forcings for p1 period
p_zwalm_p1 = p_zwalm.set_index('Timestamp').loc[p1_period]
ep_zwalm_p1 = ep_zwalm.set_index('Timestamp').loc[p1_period]
#select forcings for p2 period
p_zwalm_p2 = p_zwalm.set_index('Timestamp').loc[p2_period]
ep_zwalm_p2 = ep_zwalm.set_index('Timestamp').loc[p2_period]


# %%
lower_bound = np.array([160,0,0.1,1,0.9,0.1,0,700,0,1,0,-4.08])
upper_bound = np.array([5000,300,2,3,40,15,5000,25000,150,1.000000000000001,20,0.03])
bounds = (lower_bound, upper_bound)
print(bounds)

# %%
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
n_particles = 20
optimizer = ps.single.GlobalBestPSO(
    n_particles= n_particles, dimensions = max(parameters_initial.shape),
    options = options, bounds=bounds
)

# %%
deltat = np.single(1)
deltat_out = np.single(24)
goal_function_mNSE = lambda param: -PDM_calibration_wrapper_PSO(
    param, parameters_initial.columns, 'mNSE',p_zwalm_p1['P_thiessen'].values,
    ep_zwalm_p1['EP_thiessen'].values, area_zwalm_new, deltat,
    deltat_out, p1_period.values, p1_period_excl_warmup.values, Q_day['Value']
)

# %%
parameters_initial

# %%
cos, pos = optimizer.optimize(goal_function_mNSE, iters = 1000)

# %%
cos

# %%


# %%
pd_zwalm_out_initial = PDM(P = p_zwalm['P_thiessen'].values, 
    EP = ep_zwalm['EP_thiessen'].values,
    t = p_zwalm['Timestamp'].values,
    area = area_zwalm_initial, deltat = deltat, deltatout = deltat_out ,
    parameters = parameters_initial)
pd_zwalm_out_initial = pd_zwalm_out_initial.set_index(['Time'])

# %% [markdown]
# ## example from site

# %%
import numpy as np

# create a parameterized version of the classic Rosenbrock unconstrained optimzation function
def rosenbrock_with_args(x, a, b, c=0):
    f = (a - x[:, 0]) ** 2 + b * (x[:, 1] - x[:, 0] ** 2) ** 2 + c
    import pdb; pdb.set_trace()
    return f

# %%
x_max = 10 * np.ones(2)
x_min = -1 * x_max
bounds = (x_min, x_max)
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)

# now run the optimization, pass a=1 and b=100 as a tuple assigned to args

cost, pos = optimizer.optimize(rosenbrock_with_args, 1000, a=1, b=100, c=0)

# %%



