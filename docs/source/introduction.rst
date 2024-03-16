.. _introduction:

Installation
------------

You can install OptiMask using pip:

.. code-block:: bash

   pip install optimask

Goal of OptiMask
----------------
OptiMask is a Python package designed for efficiently handling NaN values in matrices, specifically focusing on computing the largest non-contiguous submatrix without NaN. In contrast to optimal but computationally expensive linear programming approaches, OptiMask employs a heuristic method, relying solely on Numpy for speed and efficiency. In machine learning applications, OptiMask surpasses traditional methods like Pandas ``dropna`` by maximizing the amount of valid data available for model fitting. It strategically identifies the optimal set of columns and rows to retain or remove, ensuring that the largest non-contiguous submatrix without NaN is utilized for training models.

Basic Usage
-----------

To use OptiMask, you can create an instance of the `OptiMask` class and apply the `solve` method to find the optimal rows and columnsfor a given 2D array or DataFrame. Here's a basic example:

.. code-block:: python

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

   # Print the results
   print(f"Optimal Rows: {rows}")
   print(f"Optimal Columns: {cols}")

For more detailed information on the parameters and usage, refer to the :ref:`API reference <api_reference>`.
