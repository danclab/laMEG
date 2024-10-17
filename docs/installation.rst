Windows installation
====================

Freesurfer can be ran on Windows using Microsoft Windows Subsystem for Linux (WSL).

WSL
---

Open ``cmd`` and input the following command to install WSL::

    wsl --install

Then launch ``wsl`` to install ``Freesurfer``.

Freesurfer
----------

Package Requirements
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    sudo add-apt-repository universe
    sudo apt update
    
    sudo apt -y install bc binutils libgomp1 perl psmisc sudo tar tcsh unzip uuid-dev vim-common libjpeg62-dev libxt6 libxt6-dev

Download Freesurfer 6.0.0
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.0/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz
    sudo tar -C /usr/local -xzvf freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz

Add ``FREESURFER_HOME`` and ``SUBJECT_DIR`` to your ``.bashrc``::

    export FREESURFER_HOME=/usr/local/freesurfer
    source $FREESURFER_HOME/SetUpFreeSurfer.sh
    export SUBJECTS_DIR=<path to subject data>

Download and copy your ``license.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Go to https://surfer.nmr.mgh.harvard.edu/registration.html and copy it in ``/usr/local/freesurfer``.

Enlarge your Partition, Expand Necessary Internal Space
-------------------------------------------------------

By default WSL can only use up to 50% of your PC memory, which might not be enough for ``laMEG``. For example, to execute the tutorials you would require approximately 22Gb of memory.
To increase the memory allocation for WSL (in this case, to 24GB), create a ``.wslconfig`` file in ``C:/Users/<your user name>``::

    [wsl2]
    memory=24GB

You can now proceed with the standard `laMEG installation instructions <https://github.com/danclab/laMEG/blob/main/README.rst#installation>`_.
