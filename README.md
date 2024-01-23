# laMEG
Toolbox for laminar inference with MEG

## Requirements
* Matlab
* Python version compatible with your version of Matlab: [Versions of Python Compatible with MATLAB Products by Release](https://fr.mathworks.com/support/requirements/python-compatibility.html)
* [DANC version of SPM12](https://github.com/danclab/DANC_spm12)

## Installation
Edit lameg/settings.json and change spm_path to the directory where SPM is located (e.g. /home/bonaiuto/DANC_spm12/spm12)

From the base directory:

python install.py --matlab_path /path/to/matlab/installation

For example:

python install.py --matlab_path /usr/local/MATLAB/R2018a/
