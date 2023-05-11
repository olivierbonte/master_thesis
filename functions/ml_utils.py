import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import dask
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras import models
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
from pathlib import Path
pad = Path(os.getcwd())
if pad.name == "functions":
    pad_correct = pad.parent
    os.chdir(pad_correct)
from functions.pre_processing import reshape_data, reshaped_to_train_test
from functions.plotting_functions import plot_tf_history, plot_Cstar_model


def general_sklearn_model(function, X_train, X_test, y_train, y_test,
                          t_train, t_test, Cstar, normalisation=False,seq_length = None,print_output=True, return_predictions = False, save_predictions = False, pad = None, fig = None, ax = None):
    """Wrapper around a general sklearn model to facilitate easy training, 
    predicting, scoring and visualisation

    Parameters
    ----------
    function: 
        sklearn function or pipeline. e.g. LinearRegression()
    X_train: np.ndarray
        Array with features of size (n_samples_train, n_features). Before normalisation only
    X_test: np.ndarray
        Array with features of size (n_samples_test, n_features). Before normalisation only
    y_train: np.ndarray
        Array with target of size (n_samples_train, 1)
    y_test: np.ndarray
        Array with target of size (n_samples_test, 1)
    t_train: pandas.DatetimeIndex
        Array with timestamps for training
    t_test: pandas.DatetimeIndex
        Array with timestamps for testing
    C_star: pandas.Series
        Series object with time as index and Cstar as feature
    normalisation: bool,default = False
        provides normalisation for 2D data
    seq_length: int, default = None
        if provided, data is reshaped to a window inputs with sequence length 
        seq_length
    print_output: bool, default = True
        to print both test results and output plots
    return_predictions: bool, default = False
        If true, y_train and y_test are added as ouptut
    save_predictions: bool, default = False
        If True, predictions saved as pickle  and csv to pad
    pad: pathlib.Path, default = None
        if save_predictions is True, specify the path to save to here
    fig: matplotlib.figure
        figure object 
    ax: matplotlib.axes
        axes object
        
    Returns
    -------
    func

    r2_train: numpy.float64
        R2 score on the training set
    r2_test: numpy.float
        R2 score ont the test set
    fig: matplotlib.figure
        figure object 
    ax: matplotlib.axes
        axes object
    """
    
    # t_index_train = X_train.index
    # t_index_test = X_test.index
    t_full = np.concatenate([t_train, t_test])
    # normalisation if desired
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
    #Window shaping
    if seq_length != None: 
        X_full = np.concatenate([X_train, X_test])  # type:ignore
        t_full = np.concatenate([t_train, t_test])
        y_full = np.concatenate([y_train, y_test])  # type:ignore
        X_w_full, y_w_full, t_w_full = reshape_data(
            X_full, y_full, t_full, seq_length
        )
        n_train = X_train.shape[0]  # type:ignore
        (X_train, X_test, y_train, y_test,
            t_train, t_test) = reshaped_to_train_test(
            X_w_full, y_w_full, t_w_full, seq_length, n_train, output_dim=2
        )
    
    # fitting and prediction
    func = function 
    func.fit(X_train, y_train)

    y_train_hat = func.predict(X_train)
    y_test_hat = func.predict(X_test)

    # scoring
    r2_train = r2_score(y_train, y_train_hat)
    r2_test = r2_score(y_test, y_test_hat)
    if print_output:
        print('function:' + str(function))
        print('R2 score training: ' + str(r2_train))
        print('R2 score test: ' + str(r2_test))

    # plotting
    if normalisation:
        if len(y_train_hat.shape) != 2:
            y_train_hat = y_train_hat.reshape(-1, 1)
        if len(y_test_hat.shape) != 2:
            y_test_hat = y_test_hat.reshape(-1, 1)
        y_train_hat = scaler_y.inverse_transform(y_train_hat)  # type:ignore
        y_test_hat = scaler_y.inverse_transform(y_test_hat)  # type:ignore

    if print_output:
        fig, ax = plot_Cstar_model(y_train_hat, y_test_hat, t_train, t_test, Cstar, t_full, fig, ax)
        if normalisation:
            ax.set_title('Normalised ' + str(function))# type:ignore
        else:
            ax.set_title(str(function))# type:ignore
    else:
        fig = None
        ax = None
    y_train_hat = pd.DataFrame({'C*':y_train_hat.flatten()})
    y_test_hat = pd.DataFrame({'C*':y_test_hat.flatten()})
    y_train_hat = y_train_hat.set_index(t_train)
    y_test_hat = y_test_hat.set_index(t_test)
    if save_predictions:
        if pad == None:
            raise ValueError('Provide a valid pad before setting save predictions to True')
        if not os.path.exists(pad):
            os.makedirs(pad)
        y_train_hat.to_pickle(pad/'y_train_hat.pickle')
        y_test_hat.to_pickle(pad/'y_test_hat.pickle')
        y_train_hat.to_csv(pad/'y_train_hat.csv')
        y_test_hat.to_csv(pad/'y_test_hat.csv')
    if return_predictions:
        return func, r2_train, r2_test, fig, ax, y_train_hat, y_test_hat
    else:   
        return func, r2_train, r2_test, fig, ax,


def general_tensorflow_model(model, X_train, X_test, y_train,
                             y_test, t_train, t_test, Cstar, epochs=100,
                             batch_size=32, validation_split=0.2, lstm=False,
                             normalisation=True, shuffle=False, learning_rate=1e-3,
                             training=True, print_output=True, verbose=1, seq_length=None,
                             fig=None, ax=None):
    """
    General wrapper for trainig and evaluation tensorflow models (regression)
    Trains on MSE, R2 as follow-up metric, adam optimizer.
    Default is normalisation

    Parameters
    ---------
    model: keras.Model instance
        keras model (based on tensorflow)
    X_train: np.ndarray
        Matrix with training samples (rows) and features (columns), non-normalised
    X_test: np.ndarray
        Matrix with testing samples (rows) and features (columns), non-normalised
    y_train: np.ndarray
        Vector with training targets (rows), non-normalised
    y_test: np.ndarray
        Vector with testing targets (rows), non-normalised
    t_train: pandas.DateTimeIndex
        training instances (as datetime64[ns])
    t_test: pandas.DateTimeIndex
        testing instances (as datetime64[ns])
    C_star: pandas.Series
        Series object with time as index and Cstar as feature
    epochs: int, default = 100

    batch_size: int, default =32

    validation_split: float or int, default = 0.2
        if between 0 and 1, it is interpreted as the fraction of training data withheld as validation data
        if larger than 1, it is interpreted as the number of training instances (counting from the last instance) to use for validation
        if set to 0, no validation is performed
    lstm: bool, default = False
        if True, the data is prepared for an LSTM model
    normalisation: bool, default = True
        if True, the data is normalised using the StandardScaler() from the scikit-learn library
    shuffle: bool, default = False
        if False, the validation split of (if validation_split < 1) is not shuffled.
    learning_rate: float, default = 1e-3
        learning rate to be used in the Adam optimizer
    training: bool, default = True
        if True, the model is optimized on the training data
    print_output: bool, default = True
        if True, a plot of the training history together with a plot of the predictions and targets is given
    verbose: int, default = 1
        Passed to the .fit() method of the tensorflow model. 1 displays every epoch. Set to 0 for no output when training
    seq_length: int, default = None
        length of the sequence to pass on to LSTM model, ignored when lstm = False
    fig: matplotlib.figure
        figure object to draw figure on
    ax: matplotlib.axes
        axes object to draw figure on

    Returns
    -------
    output_dict: dictionary
        number of keys in the dictionary depend on the arguments passed in the parameters
        t_train:  np.ndarray or pandas.DateTimeIndex
            training instances (as datetime64[ns]), adapted from input if LSTM
        t_test: np.ndarray or pandas.DateTimeIndex
            test instances (as datetime64[ns]), adapted from input if LSTM
        max_val_R2: float
            maximum R2 score on validation, only given if validation_split > 0
        best_epoch: int
            epoch number where the maximum validation score is reached, only given if validation_split > 0
        r2_train: float
            R2 score on training data
        r2_test: float
            R2 score on testing data
        y_train_hat: np.ndarray 
            Predicted training targets
        y_test_hat: np.ndarray 
            Predicted testing targets
        model: keras.Model instance
            if training = True, it is the optimised model
        fig: matplotlib.figure
            figure object 
        ax: matplotlib.axes
            axes object
    """
    # Make an output dict
    output_dict = {}
    if normalisation:
        y_scaler = StandardScaler()
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)
        y_scaler.fit(y_train)
        y_train = y_scaler.transform(y_train)
        y_test = y_scaler.transform(y_test)

        X_scaler = StandardScaler()
        X_scaler.fit(X_train)
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)
    t_full = np.concatenate([t_train, t_test])
    if lstm:  # needs data reshaping
        X_full = np.concatenate([X_train, X_test])  # type:ignore
        t_full = np.concatenate([t_train, t_test])
        y_full = np.concatenate([y_train, y_test])  # type:ignore
        if seq_length == None:
            raise ValueError("""Please specify a sequence length when
              training a LSTM model""")
        X_lstm_full, y_lstm_full, t_lstm_full = reshape_data(
            X_full, y_full, t_full, seq_length
        )
        n_train = X_train.shape[0]  # type:ignore
        (X_train, X_test, y_train, y_test,
         t_train, t_test) = reshaped_to_train_test(
            X_lstm_full, y_lstm_full, t_lstm_full, seq_length, n_train, output_dim=3
        )
        output_dict['t_train'] = t_train
        output_dict['t_test'] = t_test
    if (validation_split > 1) and lstm:
        X_val = X_train[-validation_split:, :, :]  # type:ignore
        X_train = X_train[:-validation_split, :, :]  # type:ignore
        y_val = y_train[-validation_split:, :]  # type:ignore
        y_train = y_train[:-validation_split, :]  # type:ignore
        val_data = (X_val, y_val)
        validation_split = 0.0  # for tensorflow internals
    elif validation_split > 1:
        raise ValueError("""Currently selecting the last n values for validation is
        only implemented for LSTM""")
    else:
        val_data = None

    output_dict['t_train'] = t_train
    output_dict['t_test'] = t_test
    if training:
        model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=tfa.metrics.RSquare()
        )
        history = model.fit(
            X_train, y_train, batch_size=batch_size,
            epochs=epochs, validation_split=validation_split,
            shuffle=shuffle, verbose=verbose, validation_data=val_data
        )

    # Determine best epoch based on validation loss
    if ((validation_split > 0) or (val_data != None)) and training:
        r2_val_loss = history.history['val_r_square']  # type:ignore
        max_loss = np.max(r2_val_loss)
        best_epoch = np.argwhere(r2_val_loss == max_loss)[0][0]
        output_dict['max_val_R2'] = max_loss
        output_dict['best_epoch'] = best_epoch

    # Prediction and scoring
    y_train_hat = model.predict(X_train)
    y_test_hat = model.predict(X_test)
    r2_train = r2_score(y_train, y_train_hat)
    r2_test = r2_score(y_test, y_test_hat)

    if normalisation:
        y_train_hat = y_scaler.inverse_transform(y_train_hat)  # type:ignore
        y_test_hat = y_scaler.inverse_transform(y_test_hat)  # type:ignore
    if print_output:
        if training:
            plot_tf_history(history, plot_object='r_square')  # type:ignore
        if (fig == None) and (ax == None):
            print('New plot generated')
            fig, ax = plt.subplots()
        Cstar[t_full].plot(ax=ax)
        ax.plot(t_train, y_train_hat, label='Train')  # type:ignore
        ax.plot(t_test, y_test_hat, label='Test')  # type:ignore
        ax.legend()  # type:ignore
        ax.set_ylabel('C* [mm]')  # type:ignore
        print(f'Training R2: {r2_train}')
        print(f'Test R2: {r2_test}')

    output_dict['r2_train'] = r2_train
    output_dict['r2_test'] = r2_test
    output_dict['y_train_hat'] = y_train_hat
    output_dict['y_test_hat'] = y_test_hat
    output_dict['model'] = model
    output_dict['fig'] = fig
    output_dict['ax'] = ax
    return output_dict


def dask_postprocess_validation(dask_out, repeats):
    """
    if general_tensorflow_model is executed multiple times with dask.delayed, the output
    is a list of dictionarnies called 'dask_out'.This processing only valid if executed with
    validation_split > 0. Also applicable to list generated without dask

    Parameters
    ----------



    Returns
    -------
    """
    epoch_list = []
    R2_val_list = []
    for i in range(repeats):
        epoch_list.append(dask_out[i]['best_epoch'])
        R2_val_list.append(dask_out[i]['max_val_R2'])
    mean_r2_val = np.mean(R2_val_list)
    recommended_nr_epochs = int(np.round(np.mean(epoch_list)))
    return mean_r2_val, recommended_nr_epochs


def dask_postprocess(dask_out, repeats):
    r2_train_list = []
    r2_test_list = []
    models_list = []
    for i in range(repeats):
        if i == 0:
            y_train_hat = dask_out[i]['y_train_hat']
            y_test_hat = dask_out[i]['y_test_hat']
        else:
            y_train_hat = np.concatenate(
                [y_train_hat, dask_out[i]['y_train_hat']],  # type:ignore
                axis=1
            )
            y_test_hat = np.concatenate(
                [y_test_hat, dask_out[i]['y_test_hat']],  # type:ignore
                axis=1
            )
        r2_train_list.append(dask_out[i]['r2_train'])
        r2_test_list.append(dask_out[i]['r2_test'])
        models_list.append(dask_out[i]['model'])

    t_train = dask_out[0]['t_train']
    t_test = dask_out[0]['t_test']
    mean_y_train_hat = np.mean(y_train_hat, axis=1)  # type:ignore
    mean_y_test_hat = np.mean(y_test_hat, axis=1)  # type:ignore
    sd_y_train_hat = np.std(y_train_hat, axis=1)  # type:ignore
    sd_y_test_hat = np.std(y_test_hat, axis=1)  # type:ignore
    mean_r2_train = np.mean(r2_train_list)
    mean_r2_test = np.mean(r2_test_list)
    return locals()

def validation_loop(model, repeats, X_train, X_test, y_train, y_test, 
                    Cstar, exec_training, pad,dask_bool = False,
                    validation_split = 0.2,**kwargs):
    """
    Loop for repeated trainig of a tensorflow model using using the `general_tensorflow_model`
    wrapper.
    Parallel execution with dask is possible, but at risk of causing overload on the memory/CPU/GPU.
    Unless otherwise specified, determines optimal number of epochs by for each iteration determining
    the epoch with the highest validation R2.
    Average of the highest validation R2's is returned. 

    Parameters
    -----------
    model: keras.Model instance
    
    X_train: pandas.DataFrame
        training features in a dataframe with time as index
    X_test: pandas.DataFrame
        test features in a dataframe with time as index
    y_train: pandas.Series
        training targets with time as index
    y_test: pandas.Series
        training targets with time as index
    Cstar: pandas.Series
        all targets with time as index
    exec_training: bool
        if True, training is executed, otherwise read from disk
        (if training previously executed)
    pad: pathlib path
        path to write results of validation to
    dask_bool: bool, default = False
        if True, parallel execution of the repeats is carried out with dask
    validation_split: float, default = 0.2  
        part of data splitted of for validation!
    **kwargs:
        keyword arguments to pass on to `general_tensorflow_model`
    
    Returns
    -------
    mean_r2_val: float
        average R2 score on validation set
    recommend_nr_epochs: int
        average of the best epochs for each validation repetition   
    
    """
    out_list = []
    if exec_training:
        for i in range(repeats):
            temp_model = models.clone_model(model)
            if dask_bool:
                delayed_result = dask.delayed(general_tensorflow_model)(#type:ignore
                    temp_model,X_train.values, X_test.values, y_train.values, 
                    y_test.values, X_train.index, X_test.index, Cstar,
                    validation_split= validation_split, **kwargs
                )
            else:
                delayed_result = general_tensorflow_model(
                    temp_model,X_train.values, X_test.values, y_train.values, 
                    y_test.values, X_train.index, X_test.index, Cstar,
                    validation_split= validation_split, **kwargs                   
                )
            out_list.append(delayed_result)
        if dask_bool:
            dask_out = dask.compute(*out_list)#type:ignore
        else:
            dask_out = out_list
        mean_r2_val, recommended_nr_epochs = dask_postprocess_validation(dask_out, repeats)
        with open(pad/'mean_r2_val.pickle', 'wb') as handle:
            pickle.dump(mean_r2_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(pad/'recommended_nr_epochs.pickle', 'wb') as handle:
            pickle.dump(recommended_nr_epochs, handle, protocol=pickle.HIGHEST_PROTOCOL)       
    else:
        with open(pad/'mean_r2_val.pickle', 'rb') as handle:
            mean_r2_val = pickle.load(handle)
        with open(pad/'recommended_nr_epochs.pickle', 'rb') as handle:
            recommended_nr_epochs = pickle.load(handle)
    print(f'Mean R2 on validation set {mean_r2_val}')
    print(f'Recommended number of epochs: {recommended_nr_epochs}')
    return mean_r2_val, recommended_nr_epochs


def full_training_loop(model, repeats, X_train, X_test, y_train, y_test,
                       Cstar, exec_training,recommended_nr_epochs, pad,
                       dask_bool = False, validation_split = 0.0, **kwargs):
    """
    Loop for repeated trainig of a tensorflow model using using the `general_tensorflow_model`
    wrapper. Parallel execution with dask. Trains on the full dataset for the optimal number 
    of epochs as determined by `validation_loop`

    Parameters
    -----------
    model: keras.Model instance
    
    X_train: pandas.DataFrame
        training features in a dataframe with time as index
    X_test: pandas.DataFrame
        test features in a dataframe with time as index
    y_train: pandas.Series
        training targets with time as index
    y_test: pandas.Series
        training targets with time as index
    exec_training: bool
        if True, training is executed, otherwise read from disk
        (if training previously executed)
    pad: pathlib path
        path to write results of validation to
    dask_bool: bool, default = False
        if True, parallel execution of the repeats is carried out with dask
    validation_split: float, default = 0.0
        As a default, no data is splitted of (trained on all data)
    **kwargs:
        keyword arguments to pass on to `general_tensorflow_model`
    
    Returns
    -------
    mean_r2_val: float
        average R2 score on validation set
    recommend_nr_epochs: int
        average of the best epochs for each validation repetition   
    out_dict: dictionary
        all local variables (except from 'models_list' and 'dask_out')
        as defined in `dask_postprocess`
    models_list: list
        list of all the trained tensorflow models
    
    """
    out_list = []
    if exec_training:
        for i in range(repeats):
            temp_model = models.clone_model(model)
            if dask_bool:
                delayed_result = dask.delayed(general_tensorflow_model)(#type:ignore
                    temp_model,X_train.values, X_test.values, y_train.values, 
                    y_test.values, X_train.index, X_test.index, Cstar,
                    recommended_nr_epochs, validation_split= validation_split,
                    **kwargs
                )
            else:
                delayed_result = general_tensorflow_model(
                    temp_model,X_train.values, X_test.values, y_train.values, 
                    y_test.values, X_train.index, X_test.index, Cstar,
                    recommended_nr_epochs, validation_split= validation_split,
                    **kwargs                   
                )
            out_list.append(delayed_result)
        if dask_bool:
            dask_out_optimised = dask.compute(*out_list)#type:ignore
        else:
            dask_out_optimised = out_list
        out_dict = dask_postprocess(dask_out_optimised, repeats)
        models_list = out_dict['models_list']
        del out_dict['models_list'] #do not save tf models as pickle file!
        del out_dict['dask_out'] #do not save tf models as pickle file!
        print(out_dict.keys())
        with open(pad/'list_optimized_models.pickle', 'wb') as handle:
            pickle.dump(out_dict, handle)
        for (i,model) in enumerate(models_list):
            name = 'model_' + str(i)
            model.save(pad/name)
    else:
        with open(pad/'list_optimized_models.pickle', 'rb') as handle:
            out_dict = pickle.load(handle)
        models_list = []
        for i in range(repeats):
            name = 'model_' + str(i)
            models_list.append(models.load_model(pad/name))
    mean_r2_train = out_dict['mean_r2_train']
    mean_r2_test = out_dict['mean_r2_test']
    print(f'Aveage R2 on training set: {mean_r2_train}')
    print(f'Average R2 on test set: {mean_r2_test}')
    return mean_r2_train, mean_r2_test, out_dict, models_list
    