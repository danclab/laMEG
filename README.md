<img src="./lameg/assets/logo.png?raw=true" alt="laMEG" title="Title" width="300"/>

Toolbox for laminar inference with MEG, powered by [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki) and [SPM](https://github.com/spm/)

[![Unit tests](https://github.com/danclab/laMEG/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/danclab/laMEG/actions/workflows/python-package-conda.yml)
[![codecov](https://codecov.io/gh/danclab/laMEG/branch/main/graph/badge.svg)](https://codecov.io/gh/danclab/laMEG)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![](https://img.shields.io/badge/Python-3.7-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Repo Size](https://img.shields.io/github/repo-size/danclab/laMEG)

## Operating system
* Windows: not supported by FreeSurfer
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

3. To install laMEG, run:

       pip install lameg

4. Before using, deactivate and reactivate the environment for changes to environment variables to take effect:

       conda deactivate
       conda activate <env name>
