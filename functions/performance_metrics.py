import warnings
import numpy as np

def NSE(Qmod,Qobs):
    """
    Calculate the Nash-Sutcliffe Efficiency of a hydrological model.
    Only calculates for timestamps where no NaN values are present in the observed flows.

    Parameters
    ----------
    Qmod: numpy.array
        modelled flows
    Qobs: numpy.array
        observed flows

    Returns
    -------
    nse: 
        Nash-Sutcliffe Efficiency


    """
    if len(Qmod) != len(Qobs):
        raise ValueError("Lenghts of Qmod and Qobs are not the same, check inputs")
    if any(np.isnan(Qobs)):
        warnings.warn("There are NaN values present in Qobs, which is not desired")
    nan_bool = np.isnan(Qobs)
    Qmod_nonan = Qmod[~nan_bool]
    Qobs_nonan = Qobs[~nan_bool]
    Qobs_avergae = np.mean(Qobs_nonan)
    T = (Qobs_nonan - Qmod_nonan)**2
    N = (Qobs_nonan - Qobs_avergae)**2
    nse = 1 - np.sum(T)/np.sum(N)
    return nse

def mNSE(Qmod,Qobs):
    """
    Calculate the modified Nash-Sutcliffe Efficiency of a hydrological model.
    This does not square the residuals, but takes the absolute value of them. 
    Only calculates for timestamps where no NaN values are present in the observed flows.

    Parameters
    ----------
    Qmod: numpy.array
        modelled flows
    Qobs: numpy.array
        observed flows

    Returns
    -------
    mnse: 
        modified Nash-Sutcliffe Efficiency


    """
    if len(Qmod) != len(Qobs):
        raise ValueError("Lenghts of Qmod and Qobs are not the same, check inputs")
    if any(np.isnan(Qobs)):
        warnings.warn("There are NaN values present in Qobs, which is not desired")
    nan_bool = np.isnan(Qobs)
    Qmod_nonan = Qmod[~nan_bool]
    Qobs_nonan = Qobs[~nan_bool]
    Qobs_avergae = np.mean(Qobs_nonan)
    T = np.abs(Qobs_nonan - Qmod_nonan)
    N = np.abs(Qobs_nonan - Qobs_avergae)
    mnse = 1 - np.sum(T)/np.sum(N)
    return mnse