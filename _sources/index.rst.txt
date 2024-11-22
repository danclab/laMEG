|Logo| ``laMEG``: laminar inference with MEG
============================================

Introduction
------------

Toolbox for laminar inference with MEG, powered by FreeSurfer (https://surfer.nmr.mgh.harvard.edu/fswiki) and SPM (https://github.com/spm/) for 
**Python 3.7**. 

The source code of the project is hosted on Github at the following address:
https://github.com/danclab/laMEG

To get started, follow the `installation instructions <https://danclab.github.io/laMEG/#installation>`_.

Available modules
-----------------

Here is a list of the modules available in ``laMEG``:

.. currentmodule:: lameg

.. toctree::
   :maxdepth: 1

.. autosummary::
   :caption: laMEG

   ~lameg.invert
   ~lameg.laminar
   ~lameg.surf
   ~lameg.viz
   ~lameg.simulate

Tutorials
----------------

A collection of tutorials is available:

.. toctree::
   :maxdepth: 2

   auto_tutorials/index

Operating system
----------------
* Windows: Follow instructions `here <https://danclab.github.io/laMEG/windows_installation.html>`_
* Mac: May work, not tested
* Linux: Tested on Ubuntu and Debian

Requirements
----------------
* `FreeSurfer v6.0 <https://surfer.nmr.mgh.harvard.edu/fswiki/rel6downloads>`_
* Python version 3.7
* Anaconda (or miniconda)

Installation
----------------
1. Create a conda environment::

       conda create -n <env name> python=3.7

   replacing ``<env name>`` with the name of the environment you would like to create (i.e. 'lameg', or the name of your project)

2. Activate the environment::

       conda activate <env name>

   replacing ``<env name>`` with name of the environment you created.

3. Install FreeSurfer, following the instructions `here <https://danclab.github.io/laMEG/freesurfer_installation.html>`_

4. To install ``laMEG``, run::

       pip install lameg

   This also installs SPM standalone and Matlab runtime, which can take some time depending on your connection speed.

5. Before using, deactivate and reactivate the environment for changes to environment variables to take effect::

       conda deactivate
       conda activate <env name>

6. If you want to run the tutorials, download and extract the `test data <https://osf.io/mgz9q/download>`_

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Funding
-------
*Supported by the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme grant agreement 864550, and a seed grant from the Fondation pour l'Audition.*

|ERC| |FPA|

.. |Logo| image:: https://github.com/danclab/laMEG/blob/main/lameg/assets/logo.png?raw=true
   :alt: laMEG
   :width: 200

.. |ERC| image:: https://github.com/danclab/laMEG/blob/main/lameg/assets/erc_logo.jpg?raw=true
   :alt: ERC
   :height: 100

.. |FPA| image:: https://github.com/danclab/laMEG/blob/main/lameg/assets/fpa_logo.png?raw=true
   :alt: FPA
   :height: 100

