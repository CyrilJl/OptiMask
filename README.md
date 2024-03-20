# <img src="https://raw.githubusercontent.com/CyrilJl/OptiMask/main/docs/source/_static/icon.svg" alt="Logo OptiMask" width="35" height="35"> OptiMask: Efficient NaN Data Removal in Python

[![PyPI Version](https://img.shields.io/pypi/v/optimask.svg)](https://pypi.org/project/optimask/) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/optimask.svg)](https://anaconda.org/conda-forge/optimask)

## Introduction

OptiMask is a Python package designed to facilitate the process of removing NaN (Not-a-Number) data from matrices while efficiently computing the largest (and not necessarily contiguous) submatrix without NaN values. This tool prioritizes practicality and compatibility with Numpy arrays and Pandas DataFrames.

## Key Features

- **Largest Submatrix without NaN:** OptiMask calculates the largest submatrix without NaN, enhancing data analysis accuracy.
- **Efficient Computation:** With optimized computation, OptiMask provides rapid results without undue delays.
- **Numpy and Pandas Compatibility:** OptiMask seamlessly adapts to both Numpy and Pandas data structures.

## Utilization

To employ OptiMask, install the `optimask` package via pip:

```bash
pip install optimask
```

OptiMask is also available on the conda-forge channel:

```bash
conda install -c conda-forge optimask
```

```bash
mamba install optimask
```

## Usage Example

Import the `OptiMask` class from the `optimask` package and utilize its methods for efficient data masking:

```python
from optimask import OptiMask
import numpy as np

# Create a matrix with NaN values
m = 120
n = 7
data = np.zeros(shape=(m, n))
data[24:72, 3] = np.nan
data[95, :5] = np.nan

# Solve for the largest submatrix without NaN values
rows, cols = OptiMask().solve(data)

# Calculate the ratio of non-NaN values in the result
coverage_ratio = len(rows) * len(cols) / data.size

# Check if there are any NaN values in the selected submatrix
has_nan_values = np.isnan(data[rows][:, cols]).any()

# Print or display the results
print(f"Coverage Ratio: {coverage_ratio:.2f}, Has NaN Values: {has_nan_values}")
# Output: Coverage Ratio: 0.85, Has NaN Values: False
```

## Documentation

For detailed documentation, including installation instructions, API usage, and examples, visit [OptiMask Documentation](https://optimask.readthedocs.io/en/latest/index.html).

## Repository Link

Find more about OptiMask on [GitHub](https://github.com/CyrilJl/OptiMask).

## Citation

If you use OptiMask in your research or work, please cite it:

```bibtex
@software{optimask2024,
  author = {Cyril Joly},
  title = {OptiMask: NaN Removal and Largest Submatrix Computation},
  year = {2024},
  url = {https://github.com/CyrilJl/OptiMask},
}
```
Or:

```OptiMask (2024). NaN Removal and Largest Submatrix Computation. Developed by Cyril Joly: https://github.com/CyrilJl/OptiMask```
