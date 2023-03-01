# master_thesis
Code repository for source code of my master's thesis: Updating a conceptual rainfall-runoff model based on radar observation and machine learning.

The main research ouput can be found in the Jupyter Notebook `Zwalm.ipynb`

To ensure proper working of the code, make a new conda environment from the `environment.yml` by typing follow code in the command prompt
```
conda env create -f environment.yml
```
In this new environment, you can execute the github repository by either downloading the code as ZIP-file or by copying the repository with git on your local device. 

Alternatively, here is also the possibility to open this repository in a containerised cloud environment with binder:
 [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/olivierbonte/master_thesis/HEAD)

## 1 Running the preprocessing
All source code for executing the preprocessing of data can be found under the `/preprocessing_files` folder.
### 1.1 Processing of the forcing data

All processing related to forcing data (rainfaill and potential evapotranspiration) can be run from the `/preprocessing_files/forcings_master.ipynb` notebook. For when the [pywaterinfo](https://fluves.github.io/pywaterinfo/) API does not work, the possibility is provided to download the results from `preprocessing_files\Zwalm_forcings_read_in.py` via a [Zenodo repository](https://doi.org/10.5281/zenodo.7689200). In this notebook, also preprocessing and the Zwalm shapefile is performed (data retrieved from another [Zenodo repository](https://doi.org/10.5281/zenodo.7689200))

### 1.2 Accessing data via OpenEO

For accessing satellite data, the [OpenEO platform](https://openeo.org/) is used. In short, it is an API that allows access to multiple cloud computing backends with multiple interfaces (R, Python, JavaScript and webbased) (for more info see [here](https://r-spatial.org/2016/11/29/openeo.html)). A full list of available backends can be found [here](https://hub.openeo.org/). For this dissertation, the backend provided by VITO (Terrascope) is used:

- `preprocessing_files\Sentinel1_OpenEO.ipynb` for SAR-imagery calibrated to $\sigma^0$ 
- `preprocessing_files\LAI_OpenEO.ipynb` for retrieving LAI-data over the catchment. 
- For processing to SAR-data to terrain-flattened $\gamma^0$ 

To use these computational resources, you will need to make an account at VITO. Instructions on how to do so can be found [here](https://docs.openeo.cloud/federation/#terrascope-registration). Note that this is required so that you can authenticate yourself for [OpenID Connect Authentication](https://openeo.org/documentation/1.0/python/#openid-connect-authentication). Note that even after this registration, it is a possibility that not enough computational resources will be allocated (especially for the computationally expensive $\gamma^0$). If one wishes to execute even the computationally expensive processes with OpenEO themselves, consider applying for a free trial or network of resources sponsoring through the [openEO Platform of ESA](https://openeo.cloud/). 

### 1.3 Processing spatial data to timeseries
The satellite data retrieved above will be transformed to timeseries data by taking the spatial average per landuse cateogry. To execute this processing, execute the steps in the notebook ``

## 2 Downloading the preprocessed data from Zenodo 
Because preprocessing the data requires significant computational resources, the outputs of all the processing chains in `/preprocessing_files` can also be downloaded as one dataset from Zenodo... (*fill in once this Zenodo dataset is created*)
