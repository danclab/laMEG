"""
This module configures the installation setup for the laMEG package, facilitating its installation
and post-installation steps.
"""
from setuptools import setup, find_packages

# Read the long description from the README.rst
with open('README.rst', 'r', encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='lameg',
    version='0.1.2',
    description='A toolbox for laminar inference with MEG',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='DANC lab',
    author_email='james.bonaiuto@isc.cnrs.fr',
    url='https://github.com/danclab/laMEG',
    packages=find_packages(include=['lameg', 'lameg.*']),
    include_package_data=True,
    install_requires=[
        "nibabel==3.2.2",
        "numpy==1.21.6",
        "scipy==1.5.4",
        "matplotlib==3.3.4",
        "joblib==1.1.1",
        "setuptools==59.6.0",
        "elephant==0.10.0",
        "neo==0.9.0",
        "quantities==0.13.0",
        "h5py==3.1.0",
        "mne==1.3.1",
        "notebook==6.4.10",
        "widgetsnbextension==3.6.6",
        "ipywidgets==7.8.1",
        "jupyterlab-widgets==1.1.7",
        "k3d==2.14.5",
        "numpy-stl==3.1.2"
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Unix'
    ],
    entry_points={
        'console_scripts': [
            'lameg-postinstall = lameg.postinstall:run_postinstall',
        ],
    },
)
