from setuptools import find_packages, setup

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(name='optimask',
      version='0.1.8',
      packages=find_packages(),
      install_requires=['numpy', 'pandas'],
      long_description_content_type='text/markdown',
      long_description=long_description
      )
