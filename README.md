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

# Create a sample data array with NaN values
data = np.array([[1, 2, np.nan],
                 [4, np.nan, 6],
                 [7, 8, 9]])

# Solve the optimization problem
optimized_rows, optimized_cols = OptiMask.solve(data)

# Print the optimized rows and columns
print("Optimized Rows:", optimized_rows)
print("Optimized Columns:", optimized_cols)
```

## Contributing

Contributions to `optimask` are welcome! If you find a bug, have a feature request, or want to contribute code, please feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/yourusername/optimask).

```
