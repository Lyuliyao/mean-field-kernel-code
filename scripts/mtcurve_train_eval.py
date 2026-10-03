"""Train and evaluate one learning-curve model.

Usage:
  python scripts/mtcurve_train_eval.py --task-id K          # K in 0..20
  python scripts/mtcurve_train_eval.py --m 5 --replicate 0
  python scripts/mtcurve_train_eval.py --task-id 0 --smoke  # short smoke run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtcurve import data as mtdata  # noqa: E402
from mtcurve.training import train_and_evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--replicate", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.task_id is not None:
        m, replicate = mtdata.task_grid()[args.task_id]
    else:
        if args.m is None or args.replicate is None:
            raise SystemExit("provide --task-id or both --m and --replicate")
        m, replicate = args.m, args.replicate

    kwargs = {}
    if args.smoke:
        kwargs = {
            "max_steps": 200,
            "val_every": 50,
            "output_dir": mtdata.REPO_ROOT
            / "outputs"
            / "mtcurve"
            / "smoke"
            / f"M{m:03d}_r{replicate}",
        }
    summary = train_and_evaluate(m, replicate, **kwargs)
    print(
        f"M={m} r={replicate}: best_step={summary['best_training_step']} "
        f"stop={summary['stopping_step']} early={summary['early_stopped']} "
        f"val={summary['best_validation_loss']:.5f} "
        f"train_wall={summary['training_walltime_seconds']:.0f}s "
        f"eval_wall={summary['evaluation_walltime_seconds']:.0f}s"
    )


if __name__ == "__main__":
    main()
