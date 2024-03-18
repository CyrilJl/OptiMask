.. _Optimask:

OptiMask Documentation
######################

.. toctree::
   :hidden:
   :maxdepth: 1   

   introduction
   notebook
   api_reference
   future

``OptiMask`` is a Python package designed for efficiently handling NaN values in matrices, specifically focusing on computing the largest
non-contiguous submatrix without NaN. In contrast to optimal but computationally expensive linear programming approaches, OptiMask
employs a heuristic method, relying solely on Numpy for speed and efficiency. In machine learning applications, OptiMask surpasses
traditional methods like pandas ``dropna`` by maximizing the amount of valid data available for model fitting. It strategically
identifies the optimal set of columns (features) and rows (samples) to retain or remove, ensuring that the largest (non-contiguous)
submatrix without NaN is utilized for training models.

The problem differs from the computation of the largest rectangles of 1s in a binary matrix (which can be tackled with dynamic programming)
and requires a novel approach.