import numpy as np
import scipy
import pandas as pd
import os
import warnings
import dask
from pathlib import Path
from numba import jit
from joblib import Parallel, delayed
pad = Path(os.getcwd())
if pad.name == "functions":
    pad_correct = pad.parent  # Path("../../Python")
    os.chdir(pad_correct)
from functions.performance_metrics import NSE, mNSE

# Newtonian nudging functions


def tau_weighing(delta_t_abs: np.timedelta64, tau: np.timedelta64):
    """
    Calculates the temporal weighting function for Newtonian Nudging

    Parameters
    ----------
    delta_t_abs: numpy.timedelta64
        The absolute difference in hours between hour of observation and hour of assimilation
    tau: numpy.timedelta64
        The number of hours before and after the time of observation for which to apply DA.

    Returns
    -------
    W_t: float
        temporal weighing factor
    """
    if delta_t_abs.astype('int') < tau.astype('int') / 2:
        W_t = 1
    elif delta_t_abs < tau:
        W_t = (tau.astype('int') - delta_t_abs.astype('int')) / \
            (tau.astype('int') / 2)
    else:
        W_t = 0
    return W_t


def NewtonianNudging(Cstar_min, Cstar_obs, gamma, kappa, delta_t, tau):
    """
    Implementation of Newtonian Nudging with only temporal weighing and constant observatoinal uncertainty and nudging factor for C* of PDM

    Paramters
    --------
    Cstar_min: float
        A priori model estimate of C*
    Cstar_obs: float
        Observed estimate of C* (via observatoin operator model)
    gamma: float
        Observational uncertainty (between 0 and 1)
    kappa: float
        Nudging factor (between 0 and 1)
    delta_t: numpy.timedelt64
        The difference in hours between hour of observatoin and hour of assimilation
    tau: numpy.timedelta64
        The number of hours before and after the time of observation for which to apply DA.

    Returns
    -------
    Cstar_plus:float
        A posteriori model estimate of C*
    """
    W_t = tau_weighing(np.abs(delta_t), tau)
    Cstar_plus = Cstar_min + gamma * kappa * W_t * (Cstar_obs - Cstar_min)
    return Cstar_plus

# Main PDM Code


def PDM(P: np.ndarray, EP: np.ndarray, t, area: np.float32, deltat, deltatout, parameters: pd.DataFrame, m: int = 3, DA=False, Cstar_obs=None, t_obs=None, gamma=None, kappa=None, tau=None, DA_experiment=False):
    """Probability Distributed Model from "The PDM rainfall-runoff model" by Moore (2007).
    References to equations in this paper are made with their respective equation number.
    From Appendix A, only formula referring to the Pareto Distribution are used.

    Parameters
    ----------
    P: numpy.array
        rainfall intensity [mm/h]. Preferably provided in np.float32
    EP: numpy.array
        evapotranspiration [mm/h]. Preferably provided in np.float32
    t: numpy.array, datetime64[ns]
        time stamps when P and EP are collected. Must start at 00:00 and end at 23:00 for hourly data.
    area: np.float32
        area of catchment [km^2]. Crucial that this is NOT an array because this will terminate numba compiling!
    deltat: float or int
        internal time resolution used by the model. Note that this time resolution
        should also be present in forcings (P and EP) [h]
    deltatout: float or int
        desired time resolution of the modelled output. Should be larger or equal to deltat [h]
    parameters: pandas.DataFrame
        Dataframe with the following columns (use dtype = np.float32 preferably)
        - cmax: maximum storage capacity related to Pareto distribution [mm]
        - cmin: minimum storage capacity related to Pareto distribution [mm]
        - b: exponent orf Pareto distribution controlling spatial variability of store capacity [-]
        -  be: exponent in actual evaporation rate [-]
        -  k1: time constant of first linear reservoir [h]
        -  k2: time constant of second linear reservoir
        -  kb: time constant of the baseflow [h mm^(m-1)] with m the exponent of the non linear reservoir (q = kS^m)
        - kg: grounwater recharge time constant [hour mm^(bg - 1)] with drainaige to groundwater being d_i = kg^(-1) (S(t) - S_t)^bg
        - St: soil tension storage capcity [mm]
        - bg: exponent of the recharge funtion [-] (cf formula at kg)
        - tdly: time delay [h] which shifts the hydrograph with the tdly
        - qconst: const flow representing returns/abstractions within the catchment [m^3/s]

    m: int, default = 3
        The groundwater storage is modelled as a non-linear reservoir of the form q = kb S^m.
        The default value of m is 3 (cubic), but a quadratic reservoir is also implemented
    DA: bool, default = False
        If True, data assimilation(DA) via Newtonian Nudging is exectued for C*. Extra arguments regarding the algorithm and the observational data must be provided below. If False, all arguments below can be ignored
    Cstar_obs: numpy.ndarray, default = None
        The observed C*, as calculated by the observation operator models. Only used if DA = True
    t_obs: numpy.ndarray, default = None
        Timestamps of when the observed C* occur (dtype = 'datetime64[ns]')
    gamma: float, default = None
        the observational uncertainty, as of now fixed for all timestamps (between 0 and 1)
    kappa: float, default = None
        The Nudging factor (between 0 and 1)
    tau: numpy.timedelta64, default = None
        The number of hours before and after the time of observation for which to apply DA. e.g. for 12 hours specify `np.timedelta64(12,'h')`. Always specify `h` when specifying tau
    DA_experiment: bool, default = False
        if True, experiments with updating S1 instead of C*

    Returns
    -------
    pd_out: pandas.DataFrame
        Dataframe with the following columns:
        - Time: datetime64[ns]
            - input time adapted to the desired delatout
        - qmodm3s: numpy.array, dtype = np.float32
            - modelled flow at end of the catchment [m^3/s]
        - Cstar: numpy.array, dtype = np.float32
            - Critical capacity of the soil moisture storage S1 [mm]
        - S1: numpy.array, dtype = np.float32
            - Soil moisture storage S1 [mm]
        - S3: numpy.array, dtype = np.float32
            - Groundwater storage S3 [mm]

    Note
    -----
    Model is currently only tested for an internal time resolution of 1 hour.
    Further testing needed to determine if other time resolution also work.
    For closest resemblance to original MATLAB code, set deltat_out to 24
    """
    # flatten the arrays if 2 dimensional
    if len(P.shape) == 2:
        P = P.flatten()
    if len(EP.shape) == 2:
        EP = EP.flatten()

    # check if frocings are of same length
    t_length = np.int32(len(P))
    t_length_ep = np.int32(len(EP))
    if t_length != t_length_ep:
        raise ValueError("""p and ep do not have the same length,
        Provide p and ep of the same length""")

    # Load the parameters
    cmax = parameters["cmax"][0]
    cmin = parameters["cmin"][0]
    b_param = parameters["b"][0]
    be = parameters["be"][0]
    k1 = parameters["k1"][0]
    k2 = parameters["k2"][0]
    kb = parameters["kb"][0]
    kg = parameters["kg"][0]
    bg = parameters["bg"][0]
    St = parameters["St"][0]
    # round up the time delay to integer value
    tdly = np.ceil(parameters["tdly"][0])
    qconst = parameters["qconst"][0]

    # constants used throughout the model
    # \bar{c} as defined in (9), cf. Appendix A
    Smax = (b_param * cmin + cmax) / (b_param + 1)
    if Smax == cmin:
        raise ValueError("Risk of zero division error")
    if cmax <= cmin:
        cmax = cmin + 1
        Warning(
            "cmax was smaller or equal to cmin. Therefore, cmax was adapted to cmin + 1")
    aream2 = area * 1000**2  # are from km^2 to m^2
    # for linear reservoir, see (26)
    delta1star = np.exp(-deltat / k1)
    delta2star = np.exp(-deltat / k2)
    delta1 = -(delta1star + delta2star)
    delta2 = delta1star * delta2star
    if k1 == k2:
        omega_0 = 1 - (1 + deltat / k1) * delta1star
        omega_1 = (delta1star - 1 + deltat / k1) * delta1star
    else:
        omega_0 = (k1 * (delta1star - 1) - k2 * (delta2star - 1)) / (k2 - k1)
        omega_1 = (k2 * (delta2star - 1) * delta1star - k1 *
                   (delta1star - 1) * delta2star) / (k2 - k1)

    if DA:
        t_hour = t.astype('datetime64[h]')
        t_obs = t_obs.astype('datetime64[h]')  # type:ignore

    def conditional_njit(condition):
        """Condition is DA: if True, do not use the numba compiler"""
        if condition:
            # return a dummy decorator that does nothing
            def dummy_decorator(func):
                return func
            return dummy_decorator
        else:
            def njit_with_numpy_error_model(func):
                # print('exectued with numba')
                return jit(func, nopython=True, error_model='numpy')
            return njit_with_numpy_error_model

    @jit(nopython=True, error_model='numpy')
    def core_execution(i, Eiacc, V, qd, di, Cstar, S1, S3, pi, qb, qbm3s, qs, qsm3s, qmodm3s):
        ### PROBABILITY DISTRIBUTED SOIL-MOISTURE STORAGE S1 ###
        # Update S1(t) if doing DA!
        if DA:
            S1[i - 1] = max(
                min(
                    cmin + (Smax - cmin) *
                    (1 - ((cmax - Cstar[i - 1]) /
                     (cmax - cmin))**(b_param + 1)),
                    Smax
                ),
                0
            )

        # Acutal Evaporation
        Eiacc[i] = EP[i] * (1 - ((Smax - S1[i - 1]) / Smax)**be)  # (8)
        # Drainage to groundwater
        if S1[i - 1] > St:
            di[i] = (1 / kg) * ((S1[i - 1] - St)**bg)  # (10)
        else:
            di[i] = 0
        # Net rainfall
        pi[i] = P[i] - Eiacc[i] - di[i]  # (14)

        condition = 0
        if pi[i] > 0:  # net rainfall
            condition = condition + 1

        # CODE STIJN VAN HOEY IDEE: i-1 = t, i = t + dt for the reservoirs, to be investigated later
        # if not DA:
        #     Cstar[i - 1] = max(
        #         min(
        #             cmin + (cmax - cmin) * (1 -
        #                                     ((Smax - S1[i - 1]) / (Smax - cmin))**(1 / (b_param + 1))),
        #             cmax
        #         ),
        #         0
        #     )

        # Cstar[i] = max(
        #     min(Cstar[i - 1] + pi[i] * deltat, cmax),
        #     0
        # )
        # S1[i] = max(
        #     min(
        #         cmin + (Smax - cmin) *
        #         (1 - ((cmax - Cstar[i]) / (cmax - cmin))**(b_param + 1)),
        #         Smax
        #     ),
        #     0
        # )
        # V[i] = max(pi[i] * deltat - (S1[i] - S1[i - 1]), 0)

        # S1[i] = min(max(S1[i - 1] + pi[i] * deltat - V[i], 0), Smax)
        # EINDE CODE STIJN VAN HOEY

        # if C[i-1] > cmin #Original, I believe non correct
        # #cf. Moore en bell (2002) OWN IDEA!!!
        if Cstar[i - 1] + pi[i] * deltat > cmin:
            condition = condition + 1

        if condition == 2:
            # these are only temporary Cstar and S1 values needed to calculate runoff generation!
            Cstar_t = Cstar[i - 1]
            Cstar_t_plus_deltat = Cstar[i - 1] + pi[i] * deltat  # (5)
            S1t = cmin + (Smax - cmin) * (1 - ((cmax - Cstar_t) /
                                               (cmax - cmin))**(b_param + 1))  # Appendix A
            if Cstar_t_plus_deltat < cmax:
                S1t_plus_deltat = cmin + \
                    (Smax - cmin) * (1 - ((cmax - Cstar_t_plus_deltat) /
                                          (cmax - cmin))**(b_param + 1))
                V[i] = pi[i] * deltat - \
                    (S1t_plus_deltat - S1t)  # Appendix A
            else:  # so when Cstar[i-1] + pi[i]*deltat > cmax
                V[i] = pi[i] * deltat - (Smax - S1t)  # (17)
        else:
            V[i] = 0  # so no runoff generated!
        # not sure if the following is necessary
        if V[i] < 0:
            V[i] = 0

        # Update S1
        S1[i] = S1[i - 1] + pi[i] * deltat - V[i]  # (16)
        if S1[i] > Smax:
            S1[i] = Smax
        elif S1[i] < 0:
            S1[i] = 0

        # Update Cstar: NOT based on (5) here (not 100% sure why),
        # but on Appendix A (relates S1 to C).
        Cstar[i] = cmin + (cmax - cmin) * (1 -
                                           ((Smax - S1[i]) / (Smax - cmin))**(1 / (b_param + 1)))
        if Cstar[i] > cmax:
            Cstar[i] = cmax
        elif Cstar[i] <= 0:  # Cstar can be smaller than cmin! e.g. if completely dry
            Cstar[i] = 0

        ### GROUNDWATER STORAGE S3 ###
        if m == 3:
            if S3[i - 1] > 0:
                S3[i] = S3[i - 1] - 1 / (3 * kb * S3[i - 1]**2) * (
                    np.exp(-3 * deltat * kb * S3[i - 1]**2) - 1) * (di[i] - kb * S3[i - 1]**3)  # (24)
                qb[i] = kb * S3[i]**3
            else:  # prevent zero division erro
                S3[i - 1] = 0
                qb[i] = 0
        elif m == 2:
            # Based on solution of horton izzard equation!
            # This solution is presented in Moore and Bell (2002) equations (A.3) and (A.4)
            a = m * kb**(1 / m)
            b = (m - 1) / m
            if di[i] < 0:
                Warning('Solution not valid for negative di')
            N = (1 - (qb[i - 1] / di[i])**(1 / 2))
            if N == 0:
                Warning('A zero division error almost occured,'
                        + 'Therefore an alterantive calculation was performed')
                # N = 0 -> z = oneindig -> qb[i] -> (z-1)/(1+z) = 1
                # -> qb[i] = di[i]
                qb[i] = di[i]
            else:
                z = np.exp(a * deltat * di[i]**(1 / 2)) * (
                    (1 + (qb[i - 1] / di[i])**(1 / 2)) /
                    N
                )
                if z == -1:
                    raise ValueError('will make demoninator 0')
                qb[i] = di[i] * ((z - 1) / (1 + z))**2
            # S3 easily obtained from qb[i]
            S3[i] = (qb[i] / kb)**(1 / m)
        else:
            raise ValueError(
                'm should be equal to 2 or 3 (as int), no other values implemented')

        if qb[i] < 0:
            qb[i] = 0
        qbm3s[i] = qb[i] * aream2 / (1000 * 3600)  # mm -> m, h -> s
        ### SURFACE STORAGE S2 ###
        qd[i] = V[i] / deltat  # added on own behalve
        if i > 1:
            qs[i] = -delta1 * qs[i - 1] - delta2 * qs[i - 2] + \
                omega_0 * qd[i] + omega_1 * qd[i - 1]
            qsm3s[i] = qs[i] * aream2 / (1000 * 3600)  # mm -> m, h -> s

        ### TOTAL FLOW ###
        qmodm3s[i + int(tdly)] = qbm3s[i] + qsm3s[i] + qconst
        return i, Eiacc, V, qd, di, Cstar, S1, S3, pi, qb, qbm3s, qs, qsm3s, qmodm3s

    @conditional_njit(DA)
    def loop_start():
        # dt = t_hour[1] - t_hour[0]
        # initialisations of variables:
        # actual evaporation [mm/h], cf (8)
        Eiacc = np.zeros(t_length, dtype=np.float32)
        # volume or basin direct runoff per unit area over interval [mm]
        V = np.zeros(t_length, dtype=np.float32)
        # Flow of direct runoff [mm/h]
        qd = np.zeros(t_length, dtype=np.float32)
        # drainage to groundwater storgae [mm/h]
        di = np.zeros(t_length, dtype=np.float32)
        # Critical storage capacity [mm]
        Cstar = np.zeros(t_length, dtype=np.float32)
        # Storage in soil moisture reservoir [mm]
        S1 = np.zeros(t_length, dtype=np.float32)
        # Storage in groundwater reservoir [mm]
        S3 = np.zeros(t_length, dtype=np.float32)
        pi = np.zeros(t_length, dtype=np.float32)  # Net rainfaill [mm]
        # Baseflow going from groundwater storage S3 [mm/h]
        qb = np.zeros(t_length, dtype=np.float32)
        # Baseflow going from groundwater storage S3 [m^3/s]
        qbm3s = np.zeros(t_length, dtype=np.float32)
        # Surface runoff from surface storage S2 [mm]
        qs = np.zeros(t_length, dtype=np.float32)
        # Surface runoff from surface storage S2 [m^3/s]
        qsm3s = np.zeros(t_length, dtype=np.float32)
        # Total modelled flow leaving the catchment [m^3/s]
        qmodm3s = np.zeros(
            (t_length + int(tdly / deltat), 1), dtype=np.float32)

        #################################
        # Initialisation of state variables
        ##################################
        i = 0
        imax = t_length
        S1[i] = Smax / 2  # So choice the intialise at 1/2 of max capacity
        S3[i] = 0.001  # random choice? Does not matter to much, cf burnup
        # Critical capacity
        Cstar[i] = cmin + (cmax - cmin) * (1 - ((Smax - S1[i]) /
                                                (Smax - cmin))**(1 / (b_param + 1)))  # Appendix A
        if Cstar[i] > cmax:
            Cstar[i] = cmax
        elif Cstar[i] < 0:  # e.g. when completely dry! Cstar goes below cmin!
            Cstar[i] = 0

        ######################################
        # Running model through all time steps
        ######################################

        i = i + 1
        # Note on the timesteps indexing: the model only starts using the forcings with index 1 (so the 2nd of the values). This is logical, as the rain falling between timestamp 0 and timestamp 1 is assigned to timestamp 1! So x[i-1] with forcing[i] => x[i] and y[i]! (x = state variables, y = output)
        istart = i
        for i in np.arange(istart, imax):
            i, Eiacc, V, qd, di, Cstar, S1, S3, pi, qb, qbm3s, qs, qsm3s, qmodm3s = core_execution(
                i, Eiacc, V, qd, di, Cstar, S1, S3, pi, qb, qbm3s, qs, qsm3s, qmodm3s)
            if DA:
                if DA_experiment:
                    Cstar_min = S1[i]
                else:
                    Cstar_min = Cstar[i]
                tmod_i = t_hour[i]
                t_assimilated_index = np.abs(tmod_i - t_obs).argmin()
                t_assimilated = t_obs[t_assimilated_index]  # type:ignore
                Cstar_obs_i = Cstar_obs[t_assimilated_index]  # type:ignore
                delta_t = tmod_i - t_assimilated
                Cstar_plus = NewtonianNudging(
                    Cstar_min, Cstar_obs_i, gamma, kappa, delta_t, tau)
                if DA_experiment:
                    S1[i] = Cstar_plus
                else:
                    Cstar[i] = Cstar_plus
        return qmodm3s, Cstar, S1, S3

    qmodm3s, Cstar, S1, S3 = loop_start()
    # tmod output: idea to only give output till last timestamp of input!
    freq_hour = str(deltatout) + 'H'
    tmod = pd.date_range(t[0], t[-1], freq=freq_hour)

    # Suppress mean of empty slice warning
    # Could be more efficient using pandas resampling
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # qmod output
        nan_fill = ((len(qmodm3s) // (deltatout / deltat)) + 1) * \
            deltatout / deltat - len(qmodm3s)
        qmodm3s = np.append(qmodm3s, np.ones(int(nan_fill)) * np.nan)
        qmodm3s = qmodm3s.reshape((-1, int(deltatout / deltat)))
        qmodm3s = np.nanmean(qmodm3s, axis=1)
        qmodm3s = qmodm3s[:len(tmod)]  # only select values in tmod!

        nan_fill_bis = ((len(Cstar) // (deltatout / deltat)) + 1) * \
            deltatout / deltat - len(Cstar)

        # Cstar
        Cstar = np.append(Cstar, np.ones(int(nan_fill_bis))
                          * np.nan)  # type:ignore
        Cstar = Cstar.reshape((-1, int(deltatout / deltat)))
        Cstar = np.nanmean(Cstar, axis=1)
        Cstar = Cstar[:len(tmod)]

        # S1
        S1 = np.append(S1, np.ones(int(nan_fill_bis)) * np.nan)
        S1 = S1.reshape((-1, int(deltatout / deltat)))
        S1 = np.nanmean(S1, axis=1)
        S1 = S1[:len(tmod)]

        # S3
        S3 = np.append(S3, np.ones(int(nan_fill_bis)) * np.nan)
        S3 = S3.reshape((-1, int(deltatout / deltat)))
        S3 = np.nanmean(S3, axis=1)
        S3 = S3[:len(tmod)]

    pd_out = pd.DataFrame(
        {"qmodm3s": qmodm3s,
         "Cstar": Cstar,
         "S1": S1,
         "S3": S3},
        dtype=np.float32
    )
    pd_out.insert(loc=0, column='Time', value=tmod)
    return pd_out


def PDM_calibration_wrapper(parameters: np.ndarray, columns: pd.Index,
                            performance_metric: str, P: np.ndarray, EP: np.ndarray, area: np.float32, deltat, deltatout, t_model: np.ndarray, t_calibration: np.ndarray, Qobs: pd.Series, *args, **kwargs):
    """
    Wrapper written around the PDM model to allow calibration with scipy.optimize.minimze().
    Based on NSE or mNSE

    Parameters
    ----------
    parameters: numpy.ndarray
        Values of the parameters in order as they normally appear in the dataframe of PDM
    colmuns: pandas.Index
        Names of the parameters in the order that they are given in PDM
    performance metric: string
        For current implementation, either 'NSE' or 'mNSE'
    P: numpy.ndarray
        rainfall intensity [mm/h]. Preferably provided in np.float32
    EP: numpy.ndarray
        evapotranspiration [mm/h]. Preferably provided in np.float32
    area: numpy.float32
        area of catchment [km^2]
    deltat: float or int
        internal time resolution used by the model. Note that this time resolution
        should also be present in forcings (P and EP) [h]
    deltatout: float or int
        desired time resolution of the modelled output. Should be larger or equal to deltat [h]
    t_model: np.ndarray, dtype = numpy.datetime64
        sequence of timesteps for which model will run
    t_calibration: np.ndarray, dtype = numpy.datetime64
        sequence of timesteps for which the model its performance metric will be computed
    Qobs: pd.Series
        Observatoinal flows in the desired time resolution. Should contain at least the timestaps
        of calibration

    Returns
    -------
    performance:
        Value of the chosen performance metric
    """
    parameters = parameters.reshape(1, -1)
    parameters_pd = pd.DataFrame(parameters, columns=columns)  # type:ignore
    pd_out = PDM(P=P, EP=EP, t=t_model,
                 area=area, deltat=deltat, deltatout=deltatout,
                 parameters=parameters_pd, *args, **kwargs)
    if performance_metric == 'NSE':
        metric = NSE
    elif performance_metric == 'mNSE':
        metric = mNSE
    else:
        raise ValueError('Only NSE and mNSE are defined as performance metric')
    Qmod = pd_out.set_index(
        'Time').loc[t_calibration, 'qmodm3s'].values  # type:ignore
    performance = metric(Qmod, Qobs[t_calibration].values)
    return performance


def PDM_calibration_wrapper_PSO(parameters: np.ndarray, columns: pd.Index,
                                performance_metric: str, P: np.ndarray, EP: np.ndarray, area: np.float32, deltat, deltatout, t_model: np.ndarray, t_calibration: np.ndarray, Qobs: pd.Series, dask_bool=False, *args, **kwargs):
    """
    Wrapper written around the PDM model to allow calibration with pyswarms PSO.
    Based on NSE or mNSE

    Parameters
    ----------
    parameters: numpy.ndarray
        Values of the parameters in order as they normally appear in the dataframe of PDM
    colmuns: pandas.Index
        Names of the parameters in the order that they are given in PDM
    performance metric: string
        For current implementation, either 'NSE' or 'mNSE'
    P: numpy.ndarray
        rainfall intensity [mm/h]. Preferably provided in np.float32
    EP: numpy.ndarray
        evapotranspiration [mm/h]. Preferably provided in np.float32
    area: numpy.float32
        area of catchment [km^2]
    deltat: float or int
        internal time resolution used by the model. Note that this time resolution
        should also be present in forcings (P and EP) [h]
    deltatout: float or int
        desired time resolution of the modelled output. Should be larger or equal to deltat [h]
    t_model: np.ndarray, dtype = numpy.datetime64
        sequence of timesteps for which model will run
    t_calibration: np.ndarray, dtype = numpy.datetime64
        sequence of timesteps for which the model its performance metric will be computed
    Qobs: pd.Series
        Observatoinal flows in the desired time resolution
    dask_bool: bool, default = Flase
        if True, will use dask for parallelisation of the different particles (not sure if this works properly)

    Returns
    -------
    performance:
        Value of the chosen performance metric
    """
    n_particles, n_params = parameters.shape
    if performance_metric == 'NSE':
        metric = NSE
    elif performance_metric == 'mNSE':
        metric = mNSE
    else:
        raise ValueError('Only NSE and mNSE are defined as performance metric')
    performances = np.zeros(n_particles)

    def PDM_loop(i, parameters, P, EP, t_model, area, deltat, deltatout, t_calibration, Qobs, metric):
        param_temp = parameters[i, :]  # type:ignore
        param_temp = pd.DataFrame(param_temp.reshape(1, -1))
        param_temp.columns = columns
        pd_out = PDM(P=P, EP=EP, t=t_model,
                     area=area, deltat=deltat, deltatout=deltatout,
                     parameters=param_temp, *args, **kwargs)
        Qmod = pd_out.set_index(
            'Time').loc[t_calibration, 'qmodm3s'].values  # type:ignore
        performance = metric(Qmod, Qobs[t_calibration].values)
        return performance
    performances_list = []
    performances_sub_list = []
    for i in range(n_particles):
        if not dask_bool:
            performances[i] = PDM_loop(i, parameters, P, EP, t_model, area,
                                       deltat, deltatout, t_calibration, Qobs, metric)
        else:
            delayed_result = dask.delayed(PDM_loop)(  # type:ignore
                i, parameters, P, EP, t_model, area, deltat, deltatout, t_calibration, Qobs, metric
            )
            performances_sub_list.append(delayed_result)
            if ((i % 4 == 0) and (i != 0)) or (i == n_particles - 1):  # parallelise per 4
                performances_sub = dask.compute(  # type:ignore
                    *performances_sub_list, scheduler='threads')
                for i in range(len(performances_sub)):
                    performances_list.append(performances_sub[i])
                performances_sub_list = []
    if dask_bool:
        performances = np.array(performances_list)
    # performances_list = Parallel(n_jobs=-1)(delayed(PDM_loop)(
    # i, parameters, P, EP, t_model, area, deltat, deltatout,t_calibration,Qobs, metric
    # ) for i in range(n_particles))
    return performances


def parameter_sampling(parameter_names, bounds, n_samples):
    """
    Samples parameters within a uniform distribution.
    Choice of how many parameters to sample

    Parameters
    ----------
    parameter_names : list
        names of parameters in order of the bounds
    bounds: list
        list of tuples with min and max values
    n_samples : int
        number of samples to generate

    Returns
    -------
    pd_samples: pandas.DataFrame
        dataframe with the parameter names and a sample value

    """
    np.random.seed(56)  # set a fixed random seed
    samples_dict = {}
    for i, par_name in enumerate(parameter_names):
        samples_dict[par_name] = np.random.uniform(
            low=bounds[i][0], high=bounds[i][1], size=n_samples
        )
    pd_samples = pd.DataFrame(samples_dict)
    return pd_samples


def Nelder_Mead_calibration(parameters, param_names, param_bounds, performance_metric, P, EP,
                            area, deltat, deltatout, t_model, t_calibration, Qobs, *args, **kwargs):
    """
    Neler-Mead calibration of the PDM on a desired performance metric and given calibration period.
    Nelder-Mead implementation as given by `scipy.optimize.minimize(method = 'Nelder-Mead') with xatol = 1e-4,
    fatol = 1e-4 and adaptive = False as options
    (cf https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize)

    Parameters
    ----------
    parameters: numpy.ndarray
        Values of the parameters in order as they normally appear in the dataframe of PDM
    param_names: pandas.Index
        Names of the parameters in the order that they are given in PDM
    param_bounds: list
        List of tuples with the bounds of the parameters
    performance_metric: string
        For current implementation, either 'NSE' or 'mNSE'
    P: numpy.ndarray
        rainfall intensity [mm/h]. Preferably provided in np.float32
    EP: numpy.ndarray
        evapotranspiration [mm/h]. Preferably provided in np.float32
    area: numpy.float32
        area of catchment [km^2]
    deltat: float or int
        internal time resolution used by the model. Note that this time resolution
        should also be present in forcings (P and EP) [h]
    deltatout: float or int
        desired time resolution of the modelled output. Should be larger or equal to deltat [h]
    t_model: np.ndarray, dtype = numpy.datetime64
        sequence of timesteps for which model will run
    t_calibration: np.ndarray, dtype = numpy.datetime64
        sequence of timesteps for which the model its performance metric will be computed
    Qobs: pd.Series
        Observatoinal flows in the desired time resolution. Should contain at least the timestaps
        of calibration
    *args: None
        arguments for scipy.optimize.minimize(), not for the Nelder-Mead method
    ** kwargs: None
        keyword arguments for scipy.optimize.minimize(), not for the Nelder-Mead method
    Returns
    -------
    parameters_calibrated: pd.DataFrame
        calibrated parameterset
    """
    def goal_function(param, info):
        perf_mertric = -PDM_calibration_wrapper(
            param, param_names, performance_metric, P, EP, area, deltat,
            deltatout, t_model, t_calibration, Qobs
        )
        if info['Nfeval'] % 50 == 0:
            print('Number of evaluation:' + str(info['Nfeval']))
            print(param)
        info['Nfeval'] += 1
        return perf_mertric

    parameters = parameters.flatten()  # 1 dimension for minimisation
    optimization_out = scipy.optimize.minimize(
        goal_function, parameters, method='Nelder-Mead',
        # callback = callback_personal_test,
        bounds=param_bounds,
        options={'fatol': 0.001, 'xatol': 0.001, 'adaptive': False},
        # with Adaptive = True an adpated alogrithm!  Matlab default is 1e-4. I used 1e-3
        # important to add solver specific options onder 'options'!
        args=({'Nfeval': 0},),
        *args, **kwargs
    )  # has to be (n,) for initial valeus
    return optimization_out
