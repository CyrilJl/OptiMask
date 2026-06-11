.. _future:

What's New?
###########

Version 1.4 (June 9, 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~
- faster large-array preprocessing using a parallel scan while preserving the previous row-major compact coordinate order
- lower memory pressure during preprocessing by avoiding full ``m * n`` coordinate scratch arrays on large inputs
- faster grouped height reductions with a threaded ``groupby_max`` path for large NaN coordinate arrays
- faster row/column ordering in large bounded-height cases using counting sort instead of general-purpose ``argsort``
- benchmarked on a ``100_000 x 1_000`` matrix with ``2.5%`` MAR and 5 warmed trials: median solve time reduced from about ``134 ms`` to about ``57 ms``
- added regression coverage to ensure the parallel preprocessing path matches the serial preprocessing output exactly


Version 1.3 (July 31, 2024)
~~~~~~~~~~~~~~~~~~~~~~~~~~
- drop cython for numba + various optimizations (speed and memory)
- special cases of NaNs in one row or on columns detected for faster processing

Version 1.2 (June 19, 2024)
~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``np.isnan(x).nonzero()`` replaced by ``np.unravel_index(np.flatnonzero(np.isnan(x)), x.shape)``, 2x faster
- fix bug when data inputed has only one row

Version 1.1 (May 10, 2024)
~~~~~~~~~~~~~~~~~~~~~~~~~~
- cython parts are introduced to replace bottleneck pure python implementations (`groupby_max`)


Future Developments
###################

OptiMask is committed to ongoing improvements to better serve its users. Planned future developments include:

1. **Enhanced Speed**: Further optimization to make data preprocessing tasks even faster.

2. **Compatibility with sklearn Transformers API**: Seamless integration for improved workflow and interoperability.

3. **Flexible Feature Control**: Enhanced options for users to prioritize columns (features) or rows (samples) based on specific needs.
