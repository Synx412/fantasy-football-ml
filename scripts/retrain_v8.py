from __future__ import annotations

"""One-command v8 retraining entry point.

Usage from the repository root:

    python scripts/retrain_v8.py

It downloads/builds the 2025/26 training table, trains both production artifacts,
and runs the deployment smoke test. Any failed stage stops the process.
"""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Optional existing 2025/26 merged_gw.csv; otherwise it is downloaded.",
    )
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    build = ["scripts/build_training_2025_26.py"]
    if args.raw is not None:
        build += ["--raw", str(args.raw)]
    run(*build)
    run("scripts/train_fpl_v8.py")
    if not args.skip_smoke:
        run("tests/smoke_test.py")

    print("\nV8 RETRAINING COMPLETE")
    print("Upload/commit these generated files:")
    print("  data/fpl_multitask_training_2025_26.csv")
    print("  models/fpl_multitask_bundle.joblib")
    print("  models/fpl_points_v2.joblib")
    print("  models/fpl_v8_report.json")


if __name__ == "__main__":
    main()
