import numpy as np
from setuptools import Extension, find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='optimask',
    version='1.2.2',
    packages=find_packages(),
    install_requires=['numpy', 'pandas'],
    description="OptiMask: extracting the largest (non-contiguous) submatrix without NaN",
    long_description_content_type='text/markdown',
    long_description=long_description,
    author='Cyril Joly',
    license='MIT',
    url='https://github.com/CyrilJl/optimask',
    classifiers=['License :: OSI Approved :: MIT License'],
    ext_modules=[
        Extension(
            "optimask.optimask_cython",
            sources=["optimask/optimask_cython.c"],
            include_dirs=[np.get_include()],
            extra_compile_args=['-std=c99'],
            extra_link_args=['-std=c99']
        )
    ]
)
