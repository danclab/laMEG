# laMEG
Toolbox for laminar inference with MEG

## Requirements
* [FreeSurfer v6.0](https://surfer.nmr.mgh.harvard.edu/fswiki/rel6downloads), setup and configured (`FREESURFER_HOME` set, `SetUpFreeSurfer.sh` sourced, and `SUBJECTS_DIR` set)
* Matlab
* Python version compatible with your version of Matlab: [Versions of Python Compatible with MATLAB Products by Release](https://fr.mathworks.com/support/requirements/python-compatibility.html)
* [DANC fork of SPM-dev](https://github.com/danclab/spm)

## Installation
Edit `lameg/settings.json` and change `spm_path` to the directory where SPM is located (e.g. `/usr/local/MATLAB/R2018a/`)

From the base directory, run:

    python install.py -m /matlab_path

where `/matlab_path` is the path to your installed Matlab (the path containing `extern/engines/python`). For example:

    python install.py -m /usr/local/MATLAB/R2018a/
