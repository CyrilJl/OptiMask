from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

from benchmark_optimask import generate_random
from optimask import OptiMask


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a single OptiMask solve call.")
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--cols", type=int, default=1000)
    parser.add_argument("--ratio", type=float, default=0.02)
    parser.add_argument("--data-seed", type=int, default=1000)
    parser.add_argument("--solve-seed", type=int, default=99)
    parser.add_argument("--sort", default="cumtime", choices=["cumtime", "tottime", "ncalls"])
    parser.add_argument("--limit", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x = generate_random(args.rows, args.cols, args.ratio, seed=args.data_seed)
    solver = OptiMask(random_state=args.solve_seed)
    solver.solve(x)

    profiler = cProfile.Profile()
    profiler.enable()
    solver.solve(x)
    profiler.disable()

    stats = pstats.Stats(profiler).strip_dirs().sort_stats(args.sort)
    stats.print_stats(args.limit)


if __name__ == "__main__":
    main()
