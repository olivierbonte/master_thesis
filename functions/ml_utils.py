from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def general_sklearn_model(function, X_train, X_test, y_train, y_test,
                           t_train, t_test, Cstar,normalisation = False,
                           print_output = True):
    """Wrapper around a general sklearn model to facilitate easy training, 
    predicting, scoring and visualisation

    Parameters
    ----------
    function: 
        sklearn function or pipeline. e.g. LinearRegression()
    X_train: np.ndarray
        Array with features of size (n_samples_train, n_features)
    X_test: np.ndarray
        Array with features of size (n_samples_test, n_features)
    y_train: np.ndarray
        Array with target of size (n_samples_train, 1)
    y_test: np.ndarray
        Array with target of size (n_samples_test, 1)
    t_train: np.ndarray
        Array with timestamps for training
    t_test: pandas.core.indexes.datetimes.DatetimeIndex 
        Array with timestamps for testing
    C_star: pandas.core.indexes.datetimes.DatetimeIndex
        Series object with time as index and Cstar as output
    normalisation: bool,default = False
        provides normalisation for 2D data
    print_output: bool, default = True
        to print both test results and output plots
    Returns
    -------
    func

    r2_train: numpy.float64

    r2_test: numpy.float

    fig

    ax
    
    """
    # t_index_train = X_train.index
    # t_index_test = X_test.index
    t_full = np.concatenate([t_train, t_test])
    #normalisation if desired
    if normalisation:
        print('Normalisation executed')
        scaler_X = StandardScaler()
        scaler_X.fit(X_train)
        X_train = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)

        scaler_y = StandardScaler()
        scaler_y.fit(y_train)
        y_train = scaler_y.transform(y_train)
        y_test = scaler_y.transform(y_test)
        
    #fitting and prediction
    func = function
    func.fit(X_train, y_train)

    y_train_hat = func.predict(X_train)
    y_test_hat = func.predict(X_test)

    #scoring
    r2_train = r2_score(y_train, y_train_hat)
    r2_test = r2_score(y_test, y_test_hat)
    if print_output:
        print('function:' + str(function))
        print('R2 score training: ' + str(r2_train))
        print('R2 score test: ' + str(r2_test))

    #plotting
    if normalisation:
        if len(y_train_hat.shape) != 2:
            y_train_hat = y_train_hat.reshape(-1,1)
        if len(y_test_hat.shape) != 2:
            y_test_hat = y_test_hat.reshape(-1,1)
        y_train_hat = scaler_y.inverse_transform(y_train_hat)#type:ignore
        y_test_hat = scaler_y.inverse_transform(y_test_hat)#type:ignore


    if print_output:
        fig, ax = plt.subplots()
        Cstar[t_full].plot(ax=ax)
        ax.plot(t_train, y_train_hat, label = 'Train')
        ax.plot(t_test, y_test_hat, label = 'Test')
        ax.legend()
        ax.set_ylabel('C* [mm]')
        if normalisation:
            ax.set_title('Normalised ' + str(function))
        else:
            ax.set_title(str(function))
    else:
        fig = None
        ax = None

    return func, r2_train, r2_test, fig, ax

