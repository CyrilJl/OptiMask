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
