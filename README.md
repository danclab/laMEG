# laMEG
Toolbox for laminar inference with MEG

[![Build and Test Python Package with Conda](https://github.com/danclab/laMEG/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/danclab/laMEG/actions/workflows/python-package-conda.yml)

## Operating system
* Windows: may work, but not supported
* Mac: may work, but not tested
* Linux: yes
  
## Requirements
* [FreeSurfer v6.0](https://surfer.nmr.mgh.harvard.edu/fswiki/rel6downloads), setup and configured (`FREESURFER_HOME` set, `SetUpFreeSurfer.sh` sourced, and `SUBJECTS_DIR` set)
* Python version 3.7
* Anaconda (or miniconda)

## Installation
1. Create a conda environment:

       conda create -n <env name> python=3.7

   replacing &lt;env name&gt; with the name of the environment you would like to create (i.e. 'lameg')

2. Activate the environment:

       conda activate <env name>

   replacing &lt;env name&gt; with name of the environment you created. 

3. To install laMEG, from the base directory, run:

       python install.py

   By default, the install script will download and install MATLAB Runtime R2019a Update9. If you want to do this yourself (make sure to install this exact version and to set the required environment variables: https://www.mathworks.com/help/compiler/install-the-matlab-runtime.html), you can run:

       python install.py --no_matlab_runtime

4. Before using, deactivate and reactivate the environment

       conda deactivate
       conda activate <env name>
