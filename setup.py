"""
This module configures the installation setup for the laMEG package, facilitating its installation
and post-installation steps. It should not be called directly, but rather from install.py
"""

import os
from setuptools import setup, find_packages
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
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Unix'
    ],
    url='https://github.com/danclab/laMEG',
    install_requires=install_requires,
    packages=find_packages(include=['lameg', 'lameg.*']),
    package_data={
        'lameg': [
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
