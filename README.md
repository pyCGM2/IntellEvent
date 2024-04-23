
The pyCGM2-intellevent module is currently in **an experimental phase**. This means that it is subject to frequent updates and modifications as we continue to improve its functionalities and stability.

To ensure you have the latest features and updates, we recommend installing both pyCGM2 and pyCGM2-intellevent in developer mode. This installation approach allows you to directly link the local directories of these packages to their respective remote GitHub repositories.

## what is IntellEvent

IntellEvent is an accurate and robust machine learning (ML) based gait event detection algorithm for 3D motion capturing data which automatically detects initial contact (IC) and foot off (FO) events during overground walking using a certain markerset. The underlying model was trained utilising a retrospective clinical 3D gait analysis dataset of 1211 patients and 61 healthy controls leading to 5717 trials with at least four gait events per trial determined by force plates. 
IntellEvent is only trained on ground truth (= force plate) data which ensures the quality of the training events (objective). For further information visite the publication website [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0288555).


## How to work with pyCGM2-intellevent

pyCGM2-intellevent operates on a server-request model. It processes requests containing foot marker trajectories, which are sent to a server configured to be compatible with trained data. The server then returns probabilities indicating the likelihood of specific foot contact events, such as "foot off" or "foot strike".


## installation guidelines


### install pycgm2 in developper mode

Clone [pyCGM2](https://github.com/pyCGM2/pyCGM2) and switch to its branch `beta`  

then, open an anaconda console

```bash
conda create --name pycgm310 python=3.10
conda activate pycgm310

cd C:/PATH/TO/YOUR/LOCAL/PYCGM2/FOLDER
conda install -c opensim-org opensim
conda install conda-forge::btk
pip install -e .
```

### install pyCGM2-intellevent

Clone [pyCGM2-intellevent](https://github.com/pyCGM2/IntellEvent)

open a new anaconda console (or go back to the `base` environment)

then create an **intellevent** virtual environment (**python3.7 only**) 

```bash
conda create -n intellevent python=3.7
```
go to your `pycgm2-intellevent` local folder, install depedencies specified in `requirements.txt`, and 
eventually install `pycgm2-intellevent` in developper mode

```bash
activate intellevent 
cd C:/PATH/TO/YOUR/LOCAL/PYCGM2/PYCGM2_INTELLEVENT_FOLDER
pip install -r requirements.txt
pip install -e . 
```

## Getting Started

first, run the server

open an anaconda console
```bash
activate intellevent
intellevent_server.exe 
```

load a gait trial into vicon nexus

open another anaconda console
```bash
activate pycgm310
cd C:/PATH/TO/YOUR/DATA/FOLDER 
intellevent_request.exe
```
## Contacts

 * intellevent instigator : bernhard.dumphart@fhstp.ac.at
 * pyCGM2 integration : fabien.leboeuf@chu-nantes.fr


## License
Attribution-NonCommercial-ShareAlike 4.0 International
