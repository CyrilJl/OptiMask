import os
import re

import numpy as np
from setuptools import Extension, find_packages, setup

# Function to read the version from the __init__.py file


def read_version():
    with open(os.path.join("optimask", "__init__.py"), "r") as f:
        version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Read the long description from README.md
with open("README.md", "r") as fh:
    long_description = fh.read()

# Setup configuration
setup(
    name='optimask',
    version=read_version(),  # Read version from __init__.py
    packages=find_packages(),
    install_requires=['numpy', 'pandas'],
    setup_requires=['numpy'],  # Ensure NumPy is available during build
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
            include_dirs=[np.get_include()]
        )
    ]
)
