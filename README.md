# SUTRA : Filaments in the ISM
_Filament identificaiton and characterisation framework_

---

The package can be used as a standalone web application or can be imported as Python library

## Installation


```bash
git clone https://git.sac.gov.in/destiny_doom/sutra.git 
cd sutra
pip install .

```
Note : we recommend creating a new conda environment before installation and make sure that current `pip` is coming from conda environment

```bash
conda create -n <environment-name>
conda activate <environment-name>
which pip
```

verify installation using following command:

```bash
sutraWeb
```

## Web Application: 

The streamlit based web application runs with the following command from anywhere in the terminal

```bash
sutraWeb
```

## Command Line Application

```bash
sutraTracer -c <path/to/column/density/map> -s <output-skeleton-map-file-name> -p
```

#### CLI options

```
options:
  -h, --help            show this help message and exit
  -c INPUT_FILE, --cd_file INPUT_FILE
                        Path to the input Column density map file.
  -s SKL_OUTPUT_FILE, --skl_output SKL_OUTPUT_FILE
                        Path to the output Skeleton map.
  -p MODEL_OUTPUT_FILE, --model_output MODEL_OUTPUT_FILE
                        Path to the output Skeleton map.
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold to convert model output to skeleton map
  -m MODEL, --model MODEL
                        Threshold to convert model output to skeleton map

```

## Python Package

Example Jupyter notebook using Sutra as Python package: 

```py
from sutra.pyCLI import cloud
cdfile = '<path-to-column-density-map.fits>'

tile1 = cloud(input_file = cdfile)
tile1.find_filament()

filament_props_table = tile1.all_filament_props()

```

## Notebook

```py


```