# OptiMask: Efficient Python Package for NaN Data Removal

Introducing OptiMask, a Python package designed to simplify the process of removing NaN (Not-a-Number) data from matrices while efficiently computing the largest submatrix without NaN. OptiMask focuses on providing immediate results, seamless compatibility with both Numpy arrays and Pandas DataFrames, and a user-friendly experience.

Highlights:

- NaN Data Removal: OptiMask streamlines the removal of NaN data from matrices, ensuring data integrity.
- Largest Submatrix: OptiMask computes the largest submatrix without NaN, enhancing data analysis accuracy.
- Swift Computation: With its fast computation, OptiMask swiftly generates results without unnecessary delays.
- Numpy and Pandas: Whether you use Numpy or Pandas, OptiMask adapts to your preferred data structure.
- Simplicity: OptiMask boasts a straightforward Python interface, making it easy for users of all levels to navigate.

Discover the efficiency of NaN data removal and submatrix optimization with OptiMask. Enhance your data processing workflows with a tool that prioritizes accuracy and simplicity

## Installation

You can install `optimask` using pip:

```
pip install optimask
```

## Usage

Import the `OptiMask` class from the `optimask` package and use its methods to optimize data masking:

```
from optimask import OptiMask
import numpy as np

m = 120
n = 7
data = np.zeros(shape=(m, n))
data[24:72, 3] = np.nan
data[95, :5] = np.nan

rows, cols = OptiMask.solve(data)
len(rows)*len(cols)/data.size, np.isnan(data[rows][:, cols]).any()
```

More information on the algorithm are presented in this [notebook](notebooks/Optimask.ipynb).

## Contributing

Contributions to `optimask` are welcome! If you find a bug, have a feature request, or want to contribute code, please feel free to open an issue or submit a pull request.
