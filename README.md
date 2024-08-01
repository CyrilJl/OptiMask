# <img src="https://raw.githubusercontent.com/CyrilJl/OptiMask/main/docs/source/_static/icon.svg" alt="Logo OptiMask" width="35" height="35"> OptiMask: Efficient NaN Data Removal in Python

[![PyPI Version](https://img.shields.io/pypi/v/optimask.svg)](https://pypi.org/project/optimask/) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/optimask.svg)](https://anaconda.org/conda-forge/optimask) [![Conda Downloads](https://anaconda.org/conda-forge/optimask/badges/downloads.svg)](https://anaconda.org/conda-forge/optimask) [![Documentation Status](https://img.shields.io/readthedocs/optimask?logo=read-the-docs)](https://optimask.readthedocs.io/en/latest/?badge=latest) [![Unit tests](https://github.com/CyrilJl/OptiMask/actions/workflows/pytest.yml/badge.svg)](https://github.com/CyrilJl/OptiMask/actions/workflows/pytest.yml) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/bdca34283887428488957959bc1abcc1)](https://app.codacy.com/gh/CyrilJl/OptiMask?utm_source=github.com&utm_medium=referral&utm_content=CyrilJl/OptiMask&utm_campaign=Badge_Grade)

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

The grey cells represent the NaN locations, the blue ones represent the valid data, and the red ones represent the rows and columns removed by the algorithm:

<img src="https://github.com/CyrilJl/OptiMask/blob/main/docs/source/_static/example0.png?raw=true" width="400">

OptiMask’s algorithm is useful for handling unstructured NaN patterns, as shown in the following example:

<img src="https://github.com/CyrilJl/OptiMask/blob/main/docs/source/_static/example2.png?raw=true" width="400">

## Performances
``OptiMask`` efficiently handles large matrices, delivering results within reasonable computation times:

```python
from optimask import OptiMask
import numpy as np

def generate_random(m, n, ratio):
    """Missing at random arrays"""
    arr = np.zeros((m, n))
    nan_count = int(ratio * m * n)
    indices = np.random.choice(m * n, nan_count, replace=False)
    arr.flat[indices] = np.nan
    return arr

x = generate_random(m=100_000, n=1_000, ratio=0.02)
%time rows, cols = OptiMask(verbose=True).solve(x)
>>> 	Trial 1 : submatrix of size 37094x49 (1817606 elements) found.
>>> 	Trial 2 : submatrix of size 35667x51 (1819017 elements) found.
>>> 	Trial 3 : submatrix of size 37908x48 (1819584 elements) found.
>>> 	Trial 4 : submatrix of size 37047x49 (1815303 elements) found.
>>> 	Trial 5 : submatrix of size 37895x48 (1818960 elements) found.
>>> Result: the largest submatrix found is of size 37908x48 (1819584 elements) found.
>>> CPU times: total: 172 ms
>>> Wall time: 435 ms
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
