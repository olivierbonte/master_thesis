# master_thesis

Code repository for source code of my master's thesis: **Updating a conceptual rainfall-runoff model based on radar observation and machine learning.** This work was also presented at the EGU23 General Assembly in Vienna, the abstract of which can be found [here](https://doi.org/10.5194/egusphere-egu23-8698). Both the poster (`Poster_EGU_Olivier_Bonte_22_04.pdf`) and the dissertation itself (`Thesis_EN_BW_Olivier_Bonte_..._....pdf`) can be found in the `docs` folder.   

After the software setup, code will be described in order of the chapters of the dissertation. **To guarantee reproducibility, please follow the order below when executing the notebooks, as each chapter (and its corresponding notebook(s)) depend on previous outputs.**
## <ins> Software setup 

### Local device
First make sure [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed on your local device. Next, either download this repository as ZIP or [use git to clone the repository](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository). Open the command-line interface (if the `conda.exe` is added to the global path, otherwise open the Anaconda Prompt), navigate to the folder where the `environment.yml` is located and install the new conda environment by typing:
```
conda env create -f environment.yml
```
Note that you can also use `mamba` instead of `conda` to solve the environment faster (for more info on `mamba`, see [here](https://mamba.readthedocs.io/en/latest/installation.html)). The current software environment is optimised for running TensorFlow on GPU for a Windows Native system. To enable GPU-computation on your own device, please adapt the tensorflow related packages as specified in the [tensorflow documentation](https://www.tensorflow.org/install/pip#windows-native). 

### Reproducible environemnt in the cloud
Alternatively, there is also the possibility to open this repository in a containerised cloud environment with binder:
 [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/olivierbonte/master_thesis/HEAD) 
 Launching this environment might take several minutes. This option is recommended for quick inspection of the results of Chapter 4 through 6, but not for execution of the preprocessing and downloading of the satellite data. 

## <ins>Chapter 2 and 3: Preprocessing <ins>
Choose one of the three preprocessing options below to obtain the required data. To proceed quickly, follow option 3 and skip the visualisation. 
### 1 Option 1: running the full preprocessing of the data
All source code for executing the preprocessing of data can be found under the `/preprocessing_files` folder.
#### 1.1 Processing of the forcing data

All processing related to forcing data (rainfaill and potential evaporation) can be run from the `/preprocessing_files/forcings_master.ipynb` notebook. For when the [pywaterinfo](https://fluves.github.io/pywaterinfo/) API does not work, the possibility is provided here to download the results from `preprocessing_files\Zwalm_forcings_read_in.py` via a [Zenodo repository](https://doi.org/10.5281/zenodo.7689200). In this notebook, also preprocessing of the delineation of the Zwalm is performed (data retrieved from another [Zenodo repository](https://doi.org/10.5281/zenodo.7688784))

#### 1.2 Accessing data via OpenEO

For accessing the satellite data, the [OpenEO platform](https://openeo.org/) is used. In short, it is an API that allows access to multiple cloud computing backends with multiple interfaces (R, Python, JavaScript and webbased) (for more info see [here](https://r-spatial.org/2016/11/29/openeo.html)). A full list of available backends can be found [here](https://hub.openeo.org/). For this dissertation, the backend provided by VITO (Terrascope) is used. One can execute following notebooks for data retrieval:

- `preprocessing_files/Sentinel1_OpenEO.ipynb` for SAR-imagery calibrated to $\sigma^0$ 
- `preprocessing_files/LAI_OpenEO.ipynb` for retrieving LAI-data over the catchment. 
- `preprocessing_files/Sentinel1_OpenEO_gamma0.ipynb` for processing SAR-data to terrain-flattened $\gamma^0$ 

To use these computational resources, you will need to make an account at VITO. Instructions on how to do so can be found [here](https://docs.openeo.cloud/federation/#terrascope-registration). Note that this is required so that you can authenticate yourself for [OpenID Connect Authentication](https://openeo.org/documentation/1.0/python/#openid-connect-authentication). Note that even after this registration, it is a possibility that not enough computational resources will be allocated (especially for the computationally expensive $\gamma^0$). If one wishes to execute even the computationally expensive processes with OpenEO themselves, consider applying for a free trial or network of resources sponsoring through the [openEO Platform of ESA](https://openeo.cloud/). Note that you can check the progress of your batch jobs in the web editor via [openeoHUB](https://hub.openeo.org/): connect to VITO backend by clicking `Open in openEO Web Editor`. 

Because of the potential difficulties in acquiring (all) the data via the OpenEO platform, the output of the notebooks can also be acquired via Zenodo. The repository is found [here](https://doi.org/10.5281/zenodo.7691342) and the notebook for downloading is `preprocessing_files/OpenEO_Zenodo.ipynb`. 

#### 1.3 Processing spatial data to timeseries
The satellite data retrieved above will be transformed to time series data by taking the spatial average per land use category. To execute this processing, execute the following notebook: `preprocessing_files/timeseries_master.ipynb` (which will execute `preprocessing_files/landuse_preprocess.py` and `preprocessing_files/timeseries_generation.py`)

### 2 Option 2: Downloading the API data from Zenodo and only running local processing 
Because preprocessing of the data via the API's is not alway straightforward, the options is also provided to download most of the data from Zenodo. To do so, run the `/preprocessing_files/all_preprocessing.ipynb` notebook. Be aware that downloading the satellite data cubes might take a while (approximately 14 GB of compressed data). 
### 3 Option 3: Only download the preprocessed data
This will be far the fastest option, as the satellite data itself are not downloaded, only the time series output. To execute this option, run the `preprocessing_files/all_download.ipynb`. 
### 4 Visualisation (*optional*)
The figures for Chapter 2 are created in `Figures/Chapter_data.ipynb`. For Chapter 3, they are in `Figures/Chapter_SAR.ipynb`. Note that the notebooks (especially the second one) won't fully run unless the satellite data have been download. Some extra (interactive) visualisations of the satellite data mostly are given in `Figures/Zwalm_OpenEO.ipynb`.

## <ins> Chapter 4: PDM Calibration <ins>
All calibration efforts are bundled in the `model_training_and_calibration/Final_calibration.ipynb` notebook. The parameters obtained in optimisation are added in this repository in the `data/Zwalm_PDM_parmeters` folder.  

## <ins> Chapter 5: Machine learning methods for the inverse observation operator <ins>

The `ml_observation_operator/data_exploration.ipynb` explores the relations between features and targets. Next, the three  inverse observation operators notebooks can be run, which will automatically download the trained models/hyperparameters results from the corresponding [Zenodo repository](https://doi.org/10.5281/zenodo.7973569) for faster execution:

- `ml_observation_operator/simple_models.ipynb`: notebook with all the inverse observation operators models that are not neural networks.
- `ml_observation_operator/ANN.ipynb`: notebook which covers the multilayer perceptron structure.
- `ml_observation_operator/LSTM.ipynb`: notebook which covers the long short-term memory network. 

## <ins> Chapter 6: Data assimilation with Newtonian nudging <ins>
The notebook for this final chapter is `data_assimilation/Newtonian_nudging.ipynb`. 

## <ins> Overview of Zenodo repositories
The five different Zenodo repositories related to this dissertation are summarised here:
 
1. [Minimal dataset](https://doi.org/10.5281/zenodo.7971288): contains the data that was not retrieved through the the pywaterinfo or OpenEO API, which are the *Vlaamse hydrografische atlas*, the land use map and the shapefile of the catchment. 
2. [pywaterinfo dataset](https://doi.org/10.5281/zenodo.7689200): all the data retrieved through the pywaterinfo API.
3. [OpenEO dataset](https://doi.org/10.5281/zenodo.7691342): the SAR and LAI data retrieved through the OpenEO API.
4. [Preprocessed dataset](https://doi.org/10.5281/zenodo.7973774): the time series resulting from processing the OpenEO and pywaterinfo datasets. 
5. [Inverse observation operator dataset](https://doi.org/10.5281/zenodo.7973569): dataset containing trained models and results of hyperparameter tuning. 