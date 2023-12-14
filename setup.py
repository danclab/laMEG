from setuptools import setup
from Cython.Build import cythonize

setup(
    name='csurf',
    ext_modules=cythonize("csurf.pyx"),
    zip_safe=False,
)