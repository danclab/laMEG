import os
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from setuptools.command.install import install

class CustomInstall(install):
    def run(self):
        # Run the standard install process
        install.run(self)

        # Post-installation: enable k3d for Jupyter
        os.system("jupyter nbextension install --py --user k3d")
        os.system("jupyter nbextension enable --py --user k3d")


# Extension module
extensions = [
    Extension(
        name="lameg.csurf",
        sources=["lameg/csurf.pyx"],
        # add any necessary compile-time flags here
    )
]


setup(
    name='lameg',
    version='0.1',
    author='DANC lab',
    author_email='james.bonaiuto@isc.cnrs.fr',
    description='A toolbox for laminar inference with MEG',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/danclab/laMEG',
    install_requires=open('requirements.txt').read().splitlines(),
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    package_data={
        'lameg': ['*.so', 'matlab/*.m'],
    },
    include_package_data=True,
    zip_safe=False,
    cmdclass={
        'install': CustomInstall,
    },
)
