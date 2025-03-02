<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/optimask.svg)](https://pypi.org/project/optimask/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/optimask.svg)](https://anaconda.org/conda-forge/optimask)
[![Conda Downloads](https://anaconda.org/conda-forge/optimask/badges/downloads.svg)](https://anaconda.org/conda-forge/optimask)
[![Documentation Status](https://img.shields.io/readthedocs/optimask?logo=read-the-docs)](https://optimask.readthedocs.io/en/latest/?badge=latest)
[![Unit tests](https://github.com/CyrilJl/OptiMask/actions/workflows/pytest.yml/badge.svg)](https://github.com/CyrilJl/OptiMask/actions/workflows/pytest.yml)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/bdca34283887428488957959bc1abcc1)](https://app.codacy.com/gh/CyrilJl/OptiMask?utm_source=github.com&utm_medium=referral&utm_content=CyrilJl/OptiMask&utm_campaign=Badge_Grade)

</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/CyrilJl/OptiMask/main/docs/source/_static/icon.svg" alt="Logo OptiMask" width="200" height="200">
</div>

# OptiMask: Efficient NaN Data Removal in Python

`OptiMask` is a Python package designed for efficiently handling NaN values in matrices, specifically focusing on computing the largest non-contiguous submatrix without NaN. OptiMask employs a heuristic method, relying on `numpy` and `numba` for speed and efficiency. In machine learning applications, OptiMask surpasses traditional methods like pandas `dropna` by maximizing the amount of valid data available for model fitting. It strategically identifies the optimal set of columns (features) and rows (samples) to retain or remove, ensuring that the largest (non-contiguous) submatrix without NaN is utilized for training models.

The problem differs from the computation of the largest rectangles of 1s in a binary matrix (which can be tackled with dynamic programming) and requires a novel approach. The problem also differs from this [algorithmic challenge](https://leetcode.com/problems/largest-submatrix-with-rearrangements/description/) in that it requires rearranging both columns and rows, rather than just columns.

## Key Features

- **Largest Submatrix without NaN:** OptiMask calculates the largest submatrix without NaN, enhancing data analysis accuracy.
- **Efficient Computation:** With optimized computation, OptiMask provides rapid results without undue delays.
- **Numpy, Pandas and Polars Compatibility:** OptiMask adapts to `numpy`, `pandas` and `polars` data structures.

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

<img src="https://github.com/CyrilJl/OptiMask/blob/main/docs/source/_static/example0.png?raw=true" width="400" alt="Strutured NaN">

OptiMask’s algorithm is useful for handling unstructured NaN patterns, as shown in the following example:

<img src="https://github.com/CyrilJl/OptiMask/blob/main/docs/source/_static/example2.png?raw=true" width="400" alt="Unstructured NaN">

## Performances

``OptiMask`` efficiently handles large matrices, delivering results within reasonable computation times:

```python
from optimask import OptiMask
import numpy as np

def generate_random(m, n, ratio):
    """Missing at random arrays"""
    return np.random.choice(a=[0, np.nan], size=(m, n), p=[1-ratio, ratio])

x = generate_random(m=100_000, n=1_000, ratio=0.02)
%time rows, cols = OptiMask(verbose=True).solve(x)
# CPU times: total: 609 ms
# Wall time: 191 ms
# 	Trial 1 : submatrix of size 35008x52 (1820416 elements) found.
# 	Trial 2 : submatrix of size 35579x51 (1814529 elements) found.
# 	Trial 3 : submatrix of size 37900x48 (1819200 elements) found.
# 	Trial 4 : submatrix of size 38040x48 (1825920 elements) found.
# 	Trial 5 : submatrix of size 37753x48 (1812144 elements) found.
# Result: the largest submatrix found is of size 38040x48 (1825920 elements) found.
```

## Documentation

For detailed documentation,API usage, examples and insights on the algorithm, visit [OptiMask Documentation](https://optimask.readthedocs.io/en/latest/index.html).

## Related Project: timefiller

If you're working with time series data, check out [timefiller](https://github.com/CyrilJl/TimeFiller), another Python package I developed for time series imputation. ``timefiller`` is designed to efficiently handle missing data in time series and relies heavily on ``optimask``.

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
