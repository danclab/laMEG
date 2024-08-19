|Logo| ``laMEG``: laminar inference with MEG
============================================

Introduction
------------

Toolbox for laminar inference with MEG, powered by FreeSurfer (https://surfer.nmr.mgh.harvard.edu/fswiki) and SPM (https://github.com/spm/) for 
**Python 3.7**. 

The source code of the project is hosted on Github at the following address:
https://github.com/danclab/laMEG

To get started, follow the installation instructions `in the README <https://github.com/danclab/laMEG?tab=readme-ov-file#installation>`_.

Operating system
----------------
* Windows: not supported by FreeSurfer
* Mac: may work, but not tested
* Linux: yes

Available modules
-----------------

Here is a list of the modules available in ``laMEG``:

.. currentmodule:: lameg

.. toctree::
   :maxdepth: 1

.. autosummary::
   :caption: laMEG
   :toctree: _autosummary

   lameg.invert
   lameg.laminar
   lameg.surf
   lameg.viz
   lameg.simulate

Tutorials
----------------

A collection of tutorials is available:

.. toctree::
   :maxdepth: 2

   auto_tutorials/index

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

