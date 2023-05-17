# %% Retime the SAR data to the hourly acquisition
import os
from pathlib import Path
import pandas as pd
pad = Path(os.getcwd())
if pad.name == "data_assimilation":
    pad_correct = pad.parent
    os.chdir(pad_correct)
from functions.pre_processing import retime_SAR
ML_data_pad = Path("data/Zwalm_data/ML_data")
X_full_all = pd.read_pickle(ML_data_pad / 'X_full_all.pkl')

# C* to use for assimilation
ml_obs_op_pad = Path("data/ml_obs_op_data")

# Linear Regression: full
y_linreg_hat_train = pd.read_pickle(
    ml_obs_op_pad / 'lin_reg/full_data/y_train_hat.pickle')
y_linreg_hat_test = pd.read_pickle(
    ml_obs_op_pad / 'lin_reg/full_data/y_test_hat.pickle')
y_linreg_hat = pd.concat([y_linreg_hat_train, y_linreg_hat_test])
# y_linreg_hat_train = retime_SAR(y_linreg_hat_train, X_full_all)
# y_linreg_hat_test = retime_SAR(y_linreg_hat_test, X_full_all)
y_linreg_hat = retime_SAR(y_linreg_hat, X_full_all)

# Linear Regression: no time feature
y_linreg_hat_train_nt = pd.read_pickle(
    ml_obs_op_pad / 'lin_reg/full_data_no_time/y_train_hat.pickle')
y_linreg_hat_test_nt = pd.read_pickle(
    ml_obs_op_pad / 'lin_reg/full_data_no_time/y_test_hat.pickle')
y_linreg_hat_nt = pd.concat([y_linreg_hat_train_nt, y_linreg_hat_test_nt])
y_linreg_hat_nt = retime_SAR(y_linreg_hat_nt, X_full_all)

# Linear Regression: no forest feature
y_linreg_hat_train_nf = pd.read_pickle(
    ml_obs_op_pad / 'lin_reg/full_data_no_forest/y_train_hat.pickle'
)
y_linreg_hat_test_nf = pd.read_pickle(
    ml_obs_op_pad / 'lin_reg/full_data_no_forest/y_test_hat.pickle'
)
y_linreg_hat_nf = pd.concat([y_linreg_hat_train_nf, y_linreg_hat_test_nf])
y_linreg_hat_nf = retime_SAR(y_linreg_hat_nf, X_full_all)

# Ridge regression: window no time
y_ridge_w_hat_train = pd.read_pickle(
    ml_obs_op_pad / 'ridge/window/y_train_hat.pickle')
y_ridge_w_hat_test = pd.read_pickle(
    ml_obs_op_pad / 'ridge/window/y_test_hat.pickle')
y_ridge_w_hat = pd.concat([y_ridge_w_hat_train, y_ridge_w_hat_test])
y_ridge_w_hat = retime_SAR(y_ridge_w_hat, X_full_all)
# y_ridge_w_hat_train = retime_SAR(y_ridge_w_hat_train, X_full_all)
# y_ridge_w_hat_test = retime_SAR(y_ridge_w_hat_test, X_full_all)

# Linear SVR
y_SVR_lin_hat_train = pd.read_pickle(
    ml_obs_op_pad / 'SVR/linear/y_train_hat.pickle')
y_SVR_lin_hat_test = pd.read_pickle(
    ml_obs_op_pad / 'SVR/linear/y_test_hat.pickle')
y_SVR_lin_hat = pd.concat([y_SVR_lin_hat_train, y_SVR_lin_hat_test])
y_SVR_lin_hat = retime_SAR(y_SVR_lin_hat, X_full_all)
# y_SVR_lin_hat_train = retime_SAR(y_SVR_lin_hat_train, X_full_all)
# y_SVR_lin_hat_test = retime_SAR(y_SVR_lin_hat_test, X_full_all)

# # GPR RBF
y_GPR_hat_train = pd.read_pickle(
    ml_obs_op_pad / 'GPR/y_train_hat.pickle')
y_GPR_hat_test = pd.read_pickle(
    ml_obs_op_pad / 'GPR/y_test_hat.pickle')
y_GPR_hat = pd.concat([y_GPR_hat_train, y_GPR_hat_test])
y_GPR_hat = retime_SAR(y_GPR_hat, X_full_all)
# y_GPR_hat_train = retime_SAR(y_GPR_hat_train, X_full_all)
# y_GPR_hat_test = retime_SAR(y_GPR_hat_test, X_full_all)
# %% Save retimed C* to pickle

y_linreg_hat.to_pickle(
    ml_obs_op_pad / 'lin_reg/full_data/y_hat_retimed.pickle')
y_linreg_hat_nt.to_pickle(
    ml_obs_op_pad / 'lin_reg/full_data_no_time/y_hat_retimed.pickle')
y_linreg_hat_nf.to_pickle(
    ml_obs_op_pad / 'lin_reg/full_data_no_forest/y_hat_retimed.pickle'
)
y_ridge_w_hat.to_pickle(
    ml_obs_op_pad / 'ridge/window/y_hat_retimed.pickle')
y_SVR_lin_hat.to_pickle(
    ml_obs_op_pad / 'SVR/linear/y_hat_retimed.pickle')
y_GPR_hat.to_pickle(
    ml_obs_op_pad / 'GPR/y_hat_retimed.pickle')
# y_linreg_hat_train.to_pickle(
#     ml_obs_op_pad / 'lin_reg/full_data/y_train_hat_retimed.pickle')
# y_linreg_hat_test.to_pickle(
#     ml_obs_op_pad / 'lin_reg/full_data/y_train_hat_retimed.pickle')
# y_ridge_w_hat_train.to_pickle(
#     ml_obs_op_pad / 'ridge/window/y_train_hat_retimed.pickle')
# y_ridge_w_hat_test.to_pickle(
#     ml_obs_op_pad / 'ridge/window/y_test_hat_retimed.pickle')
# y_SVR_lin_hat_train.to_pickle(
#     ml_obs_op_pad / 'SVR/linear/y_train_hat_retimed.pickle')
# y_SVR_lin_hat_test.to_pickle(
#     ml_obs_op_pad / 'SVR/linear_y_test_hat_retimed.pickle')
# y_GPR_hat_train.to_pickle(
#     ml_obs_op_pad / 'GPR/y_train_hat_retimed.pickle')
# y_GPR_hat_test.to_pickle(
#     ml_obs_op_pad / 'GPR/y_test_hat_retimed.pickle')

# %% Krijg
