import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

p = pd.read_csv(Path("../matlab/data/P.csv"),names = ['P'])
ep = pd.read_csv(Path("../matlab/data/EP.csv"), names = ['EP']) 
area = pd.read_csv(Path("../matlab/data/Area.csv"), names= ['Area'])

observations = pd.read_csv(Path("../matlab/output/Qmod.csv"), names = ["Q_mod"])

def PDM(forcings, area, parameters):
    """Probability Distributed Model from Moore (2007)

    Parameters
    ----------
    forcings: pandas.DataFrame
        Dataframe with the following columns:
            - P: hourly rainfall [mm/h]
            - EP: hourly evapotranspiration [mm/h]

    area: float
        area of catchment [km^2]
    parameters: pandas.DataFrame 
        Dataframe with the following columns:
            - cmax
            - cmin
            - b
            - be
            - k1
            - k2
            - kb
            - kg
            - St
            - bg
            - tdly 

    
    Returns
    -------
    Qmod: pandas.DataFrame
        1 column with the modelled flow [m^3/s]
    
    
    """



    return Qmod