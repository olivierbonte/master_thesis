import warnings
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
# https://www.statsmodels.org/devel/_modules/statsmodels/distributions/empirical_distribution.html#ECDF: simple source code really!


def input_check(func):
    def wrapper(Qmod, Qobs, *args, **kwargs):
        if len(Qmod) != len(Qobs):
            raise ValueError(
                "Lenghts of Qmod and Qobs are not the same, check inputs")
        if any(np.isnan(Qobs)):
            warnings.warn(
                "There are NaN values present in Qobs, which is not desired")
        # func(Qmod, Qobs, *args, **kwargs)
        return func(Qmod, Qobs, *args, **kwargs)
    return wrapper


@input_check
def NSE(Qmod, Qobs):
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
    nse: float
        Nash-Sutcliffe Efficiency

    """
    nan_bool = np.isnan(Qobs)
    Qmod_nonan = Qmod[~nan_bool]
    Qobs_nonan = Qobs[~nan_bool]
    Qobs_avergae = np.mean(Qobs_nonan)
    T = (Qobs_nonan - Qmod_nonan)**2
    N = (Qobs_nonan - Qobs_avergae)**2
    nse = 1 - np.sum(T) / np.sum(N)
    return nse


@input_check
def mNSE(Qmod, Qobs):
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
    mnse: float
        modified Nash-Sutcliffe Efficiency

    """
    nan_bool = np.isnan(Qobs)
    Qmod_nonan = Qmod[~nan_bool]
    Qobs_nonan = Qobs[~nan_bool]
    Qobs_avergae = np.mean(Qobs_nonan)
    T = np.abs(Qobs_nonan - Qmod_nonan)
    N = np.abs(Qobs_nonan - Qobs_avergae)
    mnse = 1 - np.sum(T) / np.sum(N)
    return mnse


@input_check
def FHV(Qmod, Qobs, exceedance_prob=0.02):
    """
    Calculate the percentage bias in percent bias in flow duration curve high-segment volume (FHV) as defined in Yilmaz et al. (2008)

    Parameters
    ----------
    Qmod: numpy.array
        modelled flows
    Qobs: numpy.array
        observed flows
    exceedance_prob: float, default = 0.02
        highest fraction of flows to take into account for calculation of the bias
    Returns
    --------
    fhv: float
        percentage bias in percent bias in flow duration curve high-segment volume

    """
    nan_bool = np.isnan(Qobs)
    Qmod_nonan = Qmod[~nan_bool]
    Qobs_nonan = Qobs[~nan_bool]
    Q_mod_sorted = np.sort(Qmod_nonan)
    Q_obs_sorted = np.sort(Qobs_nonan)

    CDF_obs_distribution = ECDF(Q_obs_sorted)
    CDF_obs = CDF_obs_distribution(Q_obs_sorted)
    CDF_mod_distribution = ECDF(Q_mod_sorted)
    CDF_mod = CDF_mod_distribution(Q_mod_sorted)

    Q_obs_H = Q_obs_sorted[CDF_obs > 1 - exceedance_prob]
    Q_mod_H = Q_mod_sorted[CDF_mod > 1 - exceedance_prob]
    if len(Q_obs_H) != len(Q_mod_H):
        # occurs when e.g. modesl produces a constant input
        len_obs = len(Q_obs_H)
        Q_mod_H = Q_mod_H[0:len_obs]
    fhv = 100 * np.nansum(Q_mod_H - Q_obs_H) / np.nansum(Q_obs_H)
    return fhv
