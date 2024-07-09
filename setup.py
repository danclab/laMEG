"""
This module configures the installation setup for the laMEG package, facilitating its installation
and post-installation steps. It should not be called directly, but rather from install.py
"""

import os
import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages
from setuptools.command.install import install

class CustomInstall(install):
    """
    This class customizes the installation process for the laMEG package.
    It extends the standard installation process to include post-installation
    steps, such as enabling k3d Jupyter extensions.
    """
    def run(self):
        # Run the standard installation process
        super().run()

        # Post-installation: enable k3d for Jupyter
        os.system("jupyter nbextension install --py --user k3d")
        os.system("jupyter nbextension enable --py --user k3d")


# Extension module
extensions = [
    Extension(
        name="lameg.surf",
        sources=["lameg/surf.pyx"],
        include_dirs=[numpy.get_include()],
        # add any necessary compile-time flags here
    )
]

# Read the long description from the README.md
with open('README.md', 'r', encoding="utf-8") as f:
    long_description = f.read()

# Read the requirements from the requirements.txt
with open('requirements.txt', 'r', encoding="utf-8") as f:
    install_requires = f.read().splitlines()

setup(
    name='lameg',
    version='0.2',
    author='DANC lab',
    author_email='james.bonaiuto@isc.cnrs.fr',
    description='A toolbox for laminar inference with MEG',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/danclab/laMEG',
    install_requires=install_requires,
    packages=find_packages(include=['lameg', 'lameg.*']),
    ext_modules=cythonize(extensions),
    package_data={
        'lameg': [
            '*.so',
            'matlab/*',
            'settings.json',
            'assets/*',
            'assets/big_brain_layer_thickness/*'
        ],
    },
    include_package_data=True,
    zip_safe=False,
    cmdclass={
        'install': CustomInstall,
    },
)
