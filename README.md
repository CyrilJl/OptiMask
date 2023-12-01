# OptiMask: Efficient NaN Data Removal in Python

[![PyPI Version](https://img.shields.io/pypi/v/optimask.svg)](https://pypi.org/project/optimask/)

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
rows, cols = OptiMask.solve(data)

# Calculate the ratio of non-NaN values in the result
coverage_ratio = len(rows) * len(cols) / data.size

# Check if there are any NaN values in the selected submatrix
has_nan_values = np.isnan(data[rows][:, cols]).any()

# Print or display the results
print(f"Coverage Ratio: {coverage_ratio:.2f}, Has NaN Values: {has_nan_values}")
# Output: Coverage Ratio: 0.85, Has NaN Values: False
```

## Further Information

Additional details about the algorithm are available in this [notebook](https://github.com/CyrilJl/OptiMask/blob/main/notebooks/Optimask.ipynb).

## Repository Link

Find more about OptiMask on [GitHub](https://github.com/CyrilJl/OptiMask).

## Contributions

Contributions to the `optimask` project are encouraged. For bug reports, feature requests, or code contributions, please open an issue or submit a pull request.
