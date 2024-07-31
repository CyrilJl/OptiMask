import os
import re

from setuptools import find_packages, setup


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
    version=read_version(),
    packages=find_packages(),
    url='https://optimask.readthedocs.io',
    install_requires=['numpy', 'pandas', 'numba'],
    description="OptiMask: extracting the largest (non-contiguous) submatrix without NaN",
    long_description_content_type='text/markdown',
    long_description=long_description,
    author='Cyril Joly',
    license='MIT',
    classifiers=['License :: OSI Approved :: MIT License'],
)
