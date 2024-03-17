.. _introduction:

Introduction
############

Installation
------------

You can install OptiMask using pip:

.. code-block:: bash

   pip install optimask

Optimask is also available on the conda-forge channel:

.. code-block:: bash

   conda install -c conda-forge optimask

.. code-block:: bash

   mamba install optimask


Basic Usage
-----------

To use OptiMask, you can create an instance of the ``OptiMask`` class and apply the ``solve`` method to find the optimal rows and
columnsfor a given 2D array or DataFrame. Here's a basic example:

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


Repository
----------

The source code of the package is available at `<https://github.com/CyrilJl/OptiMask>`_.