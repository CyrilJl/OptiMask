from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='optimask',
      version='0.1.12',
      packages=find_packages(),
      install_requires=['numpy', 'pandas'],
      description="OptiMask: to extract the largest submatrix without NaN",
      long_description_content_type='text/markdown',
      long_description=long_description
      )
