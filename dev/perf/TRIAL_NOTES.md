# OptiMask Performance Trials

Shared baseline for all trial branches:

- Benchmark target: the large-array scenarios from `tests/test_optimask.py::test_speed`
- Cases: `(100000, 1000)` and `(1000, 100000)`
- Ratio: `0.02`
- Benchmark script: `python dev/perf/benchmark_optimask.py --record`
- Profiling script: `python dev/perf/profile_optimask.py`
- Constraint: keep `n_tries` unchanged so comparisons stay valid

Trial log:

- `perf/tooling-base`: baseline
  - `(100000, 1000)`: `188.49 ms` mean
  - `(1000, 100000)`: `227.33 ms` mean
- `perf/trial-01-numpy-preprocess`: regression
  - Strategy: replace numba preprocessing with `np.isnan(...).nonzero()` and `np.unique(...)`
  - Result: `641.04 ms` / `647.30 ms`
- `perf/trial-02-preprocess-two-pass`: regression
  - Strategy: exact-size two-pass numba preprocess
  - Result: `238.73 ms` / `243.52 ms`
- `perf/trial-03-no-parallel-kernels`: regression
  - Strategy: remove `parallel=True` from hot numba kernels
  - Result: `218.26 ms` / `246.48 ms`
- `perf/trial-04-fused-step-kernels`: regression
  - Strategy: fuse alternating row/column update kernels
  - Result: `197.77 ms` / `245.14 ms`
- `perf/trial-05-preprocess-direct-cols`: best overall
  - Strategy: keep the fast one-pass preprocess but build `cols_with_nan` incrementally instead of reconstructing it with boolean filtering and `argsort`
  - Result: `186.94 ms` / `206.32 ms`
- `perf/trial-06-fast-solve-dispatch`: mixed
  - Strategy: cache optional `polars` import and avoid redundant `np.asarray(X)` work
  - Result: `190.44 ms` / `203.51 ms`
- `perf/trial-07-groupby-manual-compare`: regression
  - Strategy: replace `max(...)` inside `groupby_max` with a manual comparison
  - Result: `196.53 ms` / `213.05 ms`
- `perf/trial-08-manual-rectangle-scan`: close second
  - Strategy: compute the largest rectangle with a scalar scan instead of temporary arrays
  - Result: `190.44 ms` / `203.20 ms`
- `perf/trial-09-rectangle-plus-fast-dispatch`: mixed
  - Strategy: combine trial 8 with the solve-dispatch cleanup from trial 6
  - Result: `190.78 ms` / `204.17 ms`

Recommendation:

- Keep `perf/trial-05-preprocess-direct-cols` as the primary improvement branch.
- Consider cherry-picking trial 8 separately only if follow-up benchmarks on your target hardware confirm the horizontal-array gain is worth the extra code change.
