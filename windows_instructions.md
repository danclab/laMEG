# How to install laMEG with Windows

laMEG can be run on Windows using Microsoft Windows Subsystem for Linux (WSL).

## WSL
Open ``cmd`` and input the following command to install WSL:

`wsl --install`

## Enlarge your Partition, Expand Necessary Internal Space
By default, WSL can only use up to 50% of your PC's memory, which might not be enough for ``laMEG``. For example, you would need approx. 22Gb of memory to run the tutorials.
To allow WSL to use more memory (e.g. 24Gb), create a `.wslconfig` file in C:/Users/``<your user name>``.

```
[wsl2]
memory=24GB
```

Then launch ``wsl``.

## Install Package Requirements
From the bash prompt, run the following commands to install the package requirements.
```bash
sudo add-apt-repository universe

sudo apt update
 
sudo apt -y install bc binutils libgomp1 perl psmisc sudo tar tcsh unzip uuid-dev vim-common libjpeg62-dev libxt6
```

You can now proceed with the standard [laMEG installation instructions](https://github.com/danclab/laMEG/blob/main/README.rst#installation).
