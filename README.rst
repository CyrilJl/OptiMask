OptiMask: Efficient NaN Data Removal in Python
==============================================

Introduction
------------

OptiMask is a Python package designed to facilitate the process of removing NaN (Not-a-Number) data from matrices, while efficiently computing the largest submatrix without NaN values. This tool prioritizes practicality, compatibility with Numpy arrays and Pandas DataFrames, and user-friendliness.

Key Features
------------

- NaN Data Removal: OptiMask simplifies NaN data removal from matrices, preserving data integrity.
- Largest Submatrix: OptiMask calculates the largest submatrix without NaN, enhancing data analysis accuracy.
- Efficient Computation: With optimized computation, OptiMask provides rapid results without undue delays.
- Numpy and Pandas Compatibility: OptiMask seamlessly adapts to both Numpy and Pandas data structures.
- User-Friendly Interface: OptiMask offers an intuitive Python interface, ensuring accessibility for users of varying expertise.

Utilization
-----------

To employ OptiMask, install the `optimask` package via pip:

::

    pip install optimask

Usage Example
-------------

Import the `OptiMask` class from the `optimask` package and utilize its methods for efficient data masking:

.. code-block:: python

   from optimask import OptiMask
   import numpy as np

   m = 120
   n = 7
   data = np.zeros(shape=(m, n))
   data[24:72, 3] = np.nan
   data[95, :5] = np.nan

   rows, cols = OptiMask.solve(data)
   len(rows) * len(cols) / data.size, np.isnan(data[rows][:, cols]).any()
   # Output: (0.85, False)


Further Information
-------------------

Additional details about the algorithm are available in this `notebook <notebooks/Optimask.ipynb>`_.

Contributions
-------------

Contributions to the `optimask` project are encouraged. For bug reports, feature requests, or code contributions, please open an issue or submit a pull request.
