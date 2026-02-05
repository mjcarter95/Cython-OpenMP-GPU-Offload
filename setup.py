from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="gpuomp.gpu",
    sources=["src/gpuomp/gpu.pyx"],
    language="c++",
    include_dirs=[np.get_include()],
    extra_compile_args=[
        "-O3",
        "-fopenmp",
    ],
    extra_link_args=[
        "-fopenmp",
    ],
)

setup(
    ext_modules=cythonize([ext], compiler_directives={"language_level": "3"}),
)
