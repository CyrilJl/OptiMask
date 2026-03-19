from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from optimask import OptiMask


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "dev" / "perf" / "branch_registry.json"


def generate_random(m: int, n: int, ratio: float, seed: int) -> np.ndarray:
    arr = np.zeros((m, n), dtype=np.float32)
    nan_count = int(ratio * m * n)
    rng = np.random.default_rng(seed)
    indices = rng.choice(m * n, nan_count, replace=False)
    arr.flat[indices] = np.nan
    return arr


def get_branch_name() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return result.stdout.strip()


def benchmark_case(
    *,
    shape: tuple[int, int],
    ratio: float,
    data_seed: int,
    solve_seed: int,
    repeats: int,
) -> dict[str, object]:
    rows, cols = shape
    x = generate_random(rows, cols, ratio, seed=data_seed)
    solver = OptiMask(random_state=solve_seed)

    solver.solve(x)

    timings_ms: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        solver.solve(x)
        timings_ms.append(1e3 * (perf_counter() - start))

    return {
        "shape": [rows, cols],
        "ratio": ratio,
        "timings_ms": timings_ms,
        "mean_ms": float(np.mean(timings_ms)),
        "median_ms": float(np.median(timings_ms)),
        "min_ms": float(np.min(timings_ms)),
        "max_ms": float(np.max(timings_ms)),
    }


def load_registry() -> dict[str, object]:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"branches": {}}


def write_registry(registry: dict[str, object]) -> None:
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n")


def record_run(
    *,
    registry: dict[str, object],
    branch: str,
    cases: list[dict[str, object]],
    repeats: int,
    ratio: float,
    solve_seed: int,
    data_seed_base: int,
) -> None:
    branch_runs = registry.setdefault("branches", {}).setdefault(branch, [])
    branch_runs.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "repeats": repeats,
            "ratio": ratio,
            "solve_seed": solve_seed,
            "data_seed_base": data_seed_base,
            "cases": cases,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OptiMask on large arrays.")
    parser.add_argument("--ratio", type=float, default=0.02)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--solve-seed", type=int, default=99)
    parser.add_argument("--data-seed-base", type=int, default=1000)
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Shape formatted as rows,cols. Can be repeated. Defaults to test_speed shapes.",
    )
    parser.add_argument("--record", action="store_true", help="Record the run in branch_registry.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shapes = args.shape or ["100000,1000", "1000,100000"]

    cases: list[dict[str, object]] = []
    for index, item in enumerate(shapes):
        rows_str, cols_str = item.split(",")
        case = benchmark_case(
            shape=(int(rows_str), int(cols_str)),
            ratio=args.ratio,
            data_seed=args.data_seed_base + index,
            solve_seed=args.solve_seed,
            repeats=args.repeats,
        )
        cases.append(case)

    branch = get_branch_name()
    payload = {"branch": branch, "cases": cases}
    print(json.dumps(payload, indent=2))

    if args.record:
        registry = load_registry()
        record_run(
            registry=registry,
            branch=branch,
            cases=cases,
            repeats=args.repeats,
            ratio=args.ratio,
            solve_seed=args.solve_seed,
            data_seed_base=args.data_seed_base,
        )
        write_registry(registry)


if __name__ == "__main__":
    main()
