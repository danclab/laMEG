|Logo|

Toolbox for laminar inference with MEG, powered by `FreeSurfer <https://surfer.nmr.mgh.harvard.edu/fswiki>`_ and `SPM <https://github.com/spm/>`_

|PyPI version| |Unit tests| |Coverage| |Linting| |Python| |License| |Repo size| |PyPI downloads|

Operating system
================
* Windows: not supported by FreeSurfer
* Mac: may work, but not tested
* Linux: yes

Requirements
============
* `FreeSurfer v6.0 <https://surfer.nmr.mgh.harvard.edu/fswiki/rel6downloads>`_, setup and configured (``FREESURFER_HOME`` set, ``SetUpFreeSurfer.sh`` sourced, and ``SUBJECTS_DIR`` set)
* Python version 3.7
* Anaconda (or miniconda)

Installation
============
1. Create a conda environment::

       conda create -n <env name> python=3.7

   replacing ``<env name>`` with the name of the environment you would like to create (i.e. 'lameg', or the name of your project)

2. Activate the environment::

       conda activate <env name>

   replacing ``<env name>`` with name of the environment you created.

3. To install ``laMEG``, run::

       pip install lameg

4. Before using, deactivate and reactivate the environment for changes to environment variables to take effect::

       conda deactivate
       conda activate <env name>

5. If you want to run the tutorials, download and extract the `test data <https://osf.io/mgz9q/download>`_

Documentation and Tutorials
===========================
Once you have installed ``laMEG``, check out the
`example notebooks <https://github.com/danclab/laMEG/tree/main/examples>`_,
`tutorials <https://github.com/danclab/laMEG/tree/main/tutorials>`_, and
`documentation <https://danclab.github.io/laMEG/>`_.

Funding
=======
*Supported by the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme grant agreement 864550, and a seed grant from the Fondation pour l'Audition.*

|ERC| |FPA|


.. |Logo| image:: https://github.com/danclab/laMEG/blob/main/lameg/assets/logo.png?raw=true
   :alt: laMEG
   :width: 300

.. |PyPI version| image:: https://badge.fury.io/py/lameg.svg
   :target: https://badge.fury.io/py/lameg
   :alt: PyPI version

.. |Unit tests| image:: https://github.com/danclab/laMEG/actions/workflows/python-package-conda.yml/badge.svg
   :target: https://github.com/danclab/laMEG/actions/workflows/python-package-conda.yml
   :alt: Unit tests

.. |Coverage| image:: https://codecov.io/gh/danclab/laMEG/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/danclab/laMEG
   :alt: codecov

.. |Linting| image:: https://img.shields.io/badge/linting-pylint-yellowgreen
   :target: https://github.com/pylint-dev/pylint
   :alt: linting: pylint

.. |Python| image:: https://img.shields.io/badge/Python-3.7-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.7

.. |License| image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: License: GPL v3

.. |Repo size| image:: https://img.shields.io/github/repo-size/danclab/laMEG
   :alt: Repo Size

.. |PyPI downloads| image:: https://img.shields.io/pypi/dm/lameg
   :alt: PyPI - Downloads

.. |ERC| image:: https://github.com/danclab/laMEG/blob/main/lameg/assets/erc_logo.jpg?raw=true
   :alt: ERC
   :height: 100

.. |FPA| image:: https://github.com/danclab/laMEG/blob/main/lameg/assets/fpa_logo.png?raw=true
   :alt: FPA
   :height: 100

