# laMEG
Toolbox for laminar inference with MEG

## Requirements
* [FreeSurfer v6.0](https://surfer.nmr.mgh.harvard.edu/fswiki/rel6downloads), setup and configured (`FREESURFER_HOME` set, `SetUpFreeSurfer.sh` sourced, and `SUBJECTS_DIR` set)
* Python version 3.6
* Anaconda (or miniconda)

## Installation
1. Create a conda environment:

       conda create -n <env name> python=3.6.13

   replacing &lt;env name&gt; with the name of the environment you would like to create (i.e. 'lameg')

2. Activate the environment:

       conda activate <env name>

   replacing &lt;env name&gt; with name of the environment you created. 

3. To install laMEG, from the base directory, run:

       python install.py

4. Before using, deactivate and reactivate the environment

       conda deactivate
       conda activate <env name>
