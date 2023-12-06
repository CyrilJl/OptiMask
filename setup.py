from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='optimask',
    version='0.1.14',
    packages=find_packages(),
    install_requires=['numpy', 'pandas'],
    description="OptiMask: extracting the largest (non-contiguous) submatrix without NaN",
    long_description_content_type='text/markdown',
    long_description=long_description,
    author='Cyril Joly',
    license='MIT',
    url='https://github.com/CyrilJl/optimask',
    classifiers=['License :: OSI Approved :: MIT License'],
)
