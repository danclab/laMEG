# How to install LaMEG with Windows

Freesurfer can be ran on Windows using Microsoft Windows Subsystem for Linux (WSL) 

## WSL
At the command prompt type
`wsl --install`

Then launch wsl to install freesurfer

## Freesurfer
### Package Requirements
```bash
sudo add-apt-repository universe

sudo apt update
 
sudo apt -y install bc binutils libgomp1 perl psmisc sudo tar tcsh unzip uuid-dev vim-common libjpeg62-dev libxt6 libxt6-dev

```

### Download Freesurfer 6.0.0
```bash
wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.0/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz

sudo tar -C /usr/local -xzvf freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz


```

Add ``FREESURFER_HOME`` and ``SUBJECT_DIR`` to your `.bashrc`

```bash

export FREESURFER_HOME=/usr/local/freesurfer

source $FREESURFER_HOME/SetUpFreeSurfer.sh

export SUBJECTS_DIR=<path to subject data>

```

### Download and copy your `license.txt`
Go to <https://surfer.nmr.mgh.harvard.edu/registration.html> and copy it in `/usr/local/freesurfer`

## Enlarge WSL memory allocation
By default WSL can only use up to 50% of your PC memory, which might not be enough to run LaMEG tutorials.
To allow WSL to use more memory, create a `.wslconfig` file in C:/Users/``<your user name>``

```
[wsl2]
memory=24GB

```

You can now proceed with the standard [laMEG installation instructions](https://github.com/danclab/laMEG/blob/main/README.rst)
