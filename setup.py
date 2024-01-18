from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

# Define your extension module
extensions = [
    Extension(
        name="laMEG.csurf",
        sources=["csurf.pyx"],
        # add any necessary compile-time flags here
    )
]

setup(
    name='laMEG',
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
        # Include any package data files here
        'laMEG': ['*.so', 'matlab/*.m'],
    },
    include_package_data=True,
    zip_safe=False,
)
