# optimask: Data Masking Optimization Package

`optimask` is a Python package that provides tools for optimizing data masking strategies based on NaN values in a given data array or DataFrame. The package contains the `OptiMask` class, which facilitates the arrangement of NaN values to remove rows and columns from the input data while maximizing the data coverage.

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

## Contributing

Contributions to `optimask` are welcome! If you find a bug, have a feature request, or want to contribute code, please feel free to open an issue or submit a pull request.
