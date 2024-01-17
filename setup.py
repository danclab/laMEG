from setuptools import setup
from Cython.Build import cythonize

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
    ext_modules=cythonize("csurf.pyx"),
    package_data={'laMEG': ['matlab/*.m']},
    include_package_data=True,
    zip_safe=False,
)