import numpy as np
from numba import jit
import pandas as pd
import os
from pathlib import Path
import scipy
pad = Path(os.getcwd())
if pad.name != "Python":
    pad_correct = Path("../../Python")
    os.chdir(pad_correct)
from functions.performance_metrics import NSE, mNSE


def PDM(P:np.ndarray, EP:np.ndarray, t, area:np.float32, deltat, deltatout, parameters):
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
        - b: exponent orf Pareto distribution controlling spatial variability of 
            store capacity [-]
        -  be: exponent in actual evaporation rate [-]
        -  k1: time constant of first linear reservoir [h]
        -  k2: time constant of second linear reservoir
        -  kb: time constant of the baseflow [h mm^(m-1)] with m the exponent of 
            the non linear reservoir (q = kS^m)
        - kg: grounwater recharge time constant [hour mm^(bg - 1)] with drainaige to 
        groundwater being d_i = kg^(-1) (S(t) - S_t)^bg
        - St: soil tension storage capcity [mm]
        - bg: exponent of the recharge funtion [-] (cf formula at kg)
        - tdly: time delay [h] which shifts the hydrograph with the tdly
        - qconst: const flow representing returns/abstractions within the catchment [m^3/s] 

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
    #flatten the arrays if 2 dimensional
    if len(P.shape) == 2:
        P = P.flatten()
    if len(EP.shape) == 2:
        EP = EP.flatten()
    
    #check if frocings are of same length
    t_length = np.int32(len(P))
    t_length_ep = np.int32(len(EP))
    if t_length != t_length_ep:
        raise ValueError("""p and ep do not have the same length,
        Provide p and ep of the same length""")
    
    #Load the parameters
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
    tdly = np.ceil(parameters["tdly"][0]) #round up the time delay to integer value
    qconst = parameters["qconst"][0]

    #constants used throughout the model
    Smax = (b_param*cmin+cmax)/(b_param+1) #\bar{c} as defined in (9), cf. Appendix A
    aream2 = area*1000**2 #are from km^2 to m^2
    #for linear reservoir, see (26)
    delta1star = np.exp(-deltat/k1)
    delta2star = np.exp(-deltat/k2)
    delta1 = -(delta1star + delta2star)
    delta2 = delta1star*delta2star
    if k1 == k2:
        omega_0 = 1 - (1 + deltat/k1)*delta1star
        omega_1 = (delta1star - 1 + deltat/k1)*delta1star
    else:
        omega_0 = (k1*(delta1star - 1)-k2*(delta2star -1))/(k2 - k1)
        omega_1 = (k2*(delta2star - 1)*delta1star - k1*(delta1star - 1)*delta2star)/(k2-k1)
    @jit(nopython = True)
    def numba_start():
        #initialisations of variables:
        Eiacc = np.zeros(t_length, dtype = np.float32) #actual evaporation [mm/h], cf (8)
        V = np.zeros(t_length, dtype = np.float32) #volume or basin direct runoff per unit area over interval [mm]
        qd = np.zeros(t_length, dtype = np.float32) #Flow of direct runoff [mm/h]
        di = np.zeros(t_length, dtype = np.float32) #drainage to groundwater storgae [mm/h]
        Cstar = np.zeros(t_length, dtype = np.float32) #Critical storage capacity [mm]
        S1 = np.zeros(t_length, dtype = np.float32) #Storage in soil moisture reservoir [mm]
        S3 = np.zeros(t_length, dtype = np.float32) #Storage in groundwater reservoir [mm]
        pi = np.zeros(t_length, dtype = np.float32) #Net rainfaill [mm]
        qb = np.zeros(t_length, dtype = np.float32) #Baseflow going from groundwater storage S3 [mm/h]
        qbm3s = np.zeros(t_length, dtype = np.float32) #Baseflow going from groundwater storage S3 [m^3/s]
        qs = np.zeros(t_length, dtype = np.float32) #Surface runoff from surface storage S2 [mm]
        qsm3s = np.zeros(t_length, dtype = np.float32) #Surface runoff from surface storage S2 [m^3/s]
        qmodm3s = np.zeros((t_length + int(tdly/deltat),1), dtype = np.float32) #Total modelled flow leaving the catchment [m^3/s] 


        #################################
        # Initialisation of state variables
        ##################################
        i = 0
        imax = t_length 
        S1[i] = Smax/2 #So choice the intialise at 1/2 of max capacity
        S3[i] = 0.001 #random choice? Does not matter to much, cf burnup
        # Critical capacity
        Cstar[i] = cmin + (cmax - cmin)*(1 - ((Smax-S1[i])/(Smax - cmin))**(1/(b_param+1))) #Appendix A
        if Cstar[i] > cmax:
            Cstar[i] = cmax
        elif Cstar[i] < 0: #e.g. when completely dry! Cstar goes below cmin!
            Cstar[i] = 0

        ######################################
        # Running model through all time steps
        ######################################
        i = i + 1
        istart = i
        for i in np.arange(istart,imax):
            ### PROBABILITY DISTRIBUTED SOIL-MOISTURE STORAGE S1 ###
            #Acutal Evaporation
            Eiacc[i] = EP[i]*(1-((Smax - S1[i-1])/Smax)**be) # (8)
            #Drainage to groundwater
            if S1[i-1] > St:
                di[i] = (1/kg)*((S1[i-1]- St)**bg) # (10)
            else:
                di[i] = 0
            #Net rainfall
            pi[i] = P[i] - Eiacc[i] - di[i] # (14)

            condition = 0
            if pi[i] > 0: #net rainfall
                condition = condition + 1
            #if C[i-1] > cmin #Original, I believe non correct
            if Cstar[i-1] + pi[i]*deltat > cmin: #cf. Moore en bell (2002) OWN IDEA!!!
                condition = condition + 1

            if condition == 2:
                Cstar_t = Cstar[i-1]
                Cstar_t_plus_deltat = Cstar[i-1] + pi[i]*deltat #  (5)
                S1t = cmin + (Smax - cmin)*(1 - ((cmax - Cstar_t)/(cmax - cmin))**(b_param+1)) # Appendix A
                if Cstar_t_plus_deltat < cmax:
                    S1t_plus_deltat = cmin + (Smax - cmin)*(1 - ((cmax - Cstar_t_plus_deltat)/(cmax - cmin))**(b_param+1))
                    V[i] = pi[i]*deltat - (S1t_plus_deltat - S1t) #Appendix A
                else: #so when Cstar[i-1] + pi[i]*deltat > cmax
                    V[i] = pi[i]*deltat - (Smax - S1t) # (17)
            else:
                V[i] = 0 #so no runoff generated!
            #not sure if the following is necessary
            if V[i] < 0:
                V[i] = 0
            
            #Update S1
            S1[i] = S1[i-1] + pi[i]*deltat - V[i] # (16)
            if S1[i] > Smax:
                S1[i] = Smax
            elif S1[i] < 0:
                S1[i] = 0
            
            #Update Cstar: NOT based on (5) here (not 100% sure why),
            # but on Appendix A (relates S1 to C)
            Cstar[i] = cmin + (cmax - cmin)*(1 - ((Smax - S1[i])/(Smax - cmin))**(1/(b_param+1)))
            if Cstar[i] > cmax:
                Cstar[i] = cmax
            elif Cstar[i] <= 0: #Cstar can be smaller than cmin! e.g. if completely dry
                Cstar[i] = 0

            ### GROUNDWATER STORAGE S3 ###
            S3[i] = S3[i-1] - 1/(3*kb*S3[i-1]**2)*(np.exp(-3*deltat*kb*S3[i-1]**2)-1)*(di[i]-kb*S3[i-1]**3) # (24)
            qb[i] = kb*S3[i]**3
            if qb[i] < 0:
                qb[i] = 0
            qbm3s[i] = qb[i]*aream2/(1000*3600) #mm -> m, h -> s

            ### SURFACE STORAGE S2 ###
            qd[i] = V[i]/deltat #added on own behalve
            if i > 1:
                qs[i] = -delta1*qs[i-1] - delta2*qs[i-2] +omega_0*qd[i] + omega_1*qd[i-1]
                qsm3s[i] = qs[i]*aream2/(1000*3600) #mm -> m, h -> s
        
            ### TOTAL FLOW ###
            qmodm3s[i+int(tdly)] = qbm3s[i] + qsm3s[i] + qconst
        return qmodm3s, Cstar, S1, S3

    qmodm3s, Cstar, S1, S3 = numba_start()
    #tmod output: idea to only give output till last timestamp of input!
    freq_hour = str(deltatout) + 'H'
    tmod = pd.date_range(t[0],t[-1], freq = freq_hour) 

    #qmod output
    nan_fill = ((len(qmodm3s) // (deltatout/deltat)) + 1)*deltatout/deltat - len(qmodm3s)
    qmodm3s = np.append(qmodm3s, np.ones(int(nan_fill))*np.nan)
    qmodm3s = qmodm3s.reshape((-1,int(deltatout/deltat)))
    qmodm3s = np.nanmean(qmodm3s, axis = 1)
    qmodm3s = qmodm3s[:len(tmod)] #only select values in tmod!

    nan_fill_bis = ((len(Cstar) // (deltatout/deltat)) + 1)*deltatout/deltat - len(Cstar)
    #Cstar 
    Cstar = np.append(Cstar, np.ones(int(nan_fill_bis))*np.nan)
    Cstar = Cstar.reshape((-1,int(deltatout/deltat)))
    Cstar = np.nanmean(Cstar, axis = 1)
    Cstar = Cstar[:len(tmod)] 

    #S1
    S1 = np.append(S1, np.ones(int(nan_fill_bis))*np.nan)
    S1 = S1.reshape((-1,int(deltatout/deltat)))
    S1 = np.nanmean(S1, axis = 1)
    S1 = S1[:len(tmod)] 

    #S3
    S3 = np.append(S3, np.ones(int(nan_fill_bis))*np.nan)
    S3 = S3.reshape((-1,int(deltatout/deltat)))
    S3 = np.nanmean(S3, axis = 1)
    S3 = S3[:len(tmod)] 
        
    
    pd_out = pd.DataFrame(
        {"qmodm3s":qmodm3s,
        "Cstar":Cstar,
        "S1":S1,
        "S3":S3},
        dtype = np.float32
    )
    pd_out.insert(loc = 0, column = 'Time', value = tmod)
    return pd_out

def PDM_calibration_wrapper(parameters:np.ndarray, columns:pd.Index, performance_metric:str, 
P:np.ndarray, EP:np.ndarray,area:np.float32, deltat, deltatout, t_model:np.ndarray, 
 t_calibration:np.ndarray, Qobs:pd.Series):
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
        Observatoinal flows in the desired time resolution

    Returns
    -------
    performance:    
        Value of the chosen performance metric
    """
    parameters = parameters.reshape(1,-1)
    parameters = pd.DataFrame(parameters, columns= columns)#type:ignore
    pd_out = PDM(P = P, EP = EP, t = t_model, 
        area = area, deltat = deltat, deltatout = deltatout ,
        parameters = parameters)
    if performance_metric =='NSE':
        metric = NSE
    elif performance_metric == 'mNSE':
        metric = mNSE
    else:
        raise ValueError('Only NSE and mNSE are defined as performance metric')
    Qmod = pd_out.set_index('Time').loc[t_calibration,'qmodm3s'].values
    performance = metric(Qmod, Qobs[t_calibration].values)
    return performance



