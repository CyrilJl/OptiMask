from setuptools import setup, find_packages

setup(name='optimask',
      version='0.1.1',
      packages=find_packages(),
      install_requires=['numpy', 'pandas'],
      long_description_content_type='text/markdown',
      long_description="`OptiMask` is a cutting-edge Python package designed to optimize data masking strategies by intelligently arranging NaN values in a way that maximizes the data coverage, ensuring that the number of cells available for analysis is maximized."
      )
