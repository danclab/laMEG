# How to install FreeSurfer

laMEG requires FreeSurfer to process cortical surfaces.

## Install Package Requirements
From the bash prompt, run the following commands to install FreeSurfer's package requirements.
```bash
sudo add-apt-repository universe

sudo apt update
 
sudo apt -y install bc binutils libgomp1 perl psmisc sudo tar tcsh unzip uuid-dev vim-common libjpeg62-dev libxt6 libxt6-dev
```

## Download Freesurfer 6.0.0
```bash
wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.0/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz

sudo tar -C /usr/local -xzvf freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz
```

Add ``FREESURFER_HOME`` and ``SUBJECT_DIR`` to /home/``<your username>``/.bashrc :

```bash
export FREESURFER_HOME=/usr/local/freesurfer

source $FREESURFER_HOME/SetUpFreeSurfer.sh

export SUBJECTS_DIR=<path to subject data>
```

## Add a `license.txt` File
Go to <https://surfer.nmr.mgh.harvard.edu/registration.html> and copy it in `/usr/local/freesurfer`.