"""Run the locked final model comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .experiment_config import FROZEN_EPOCHS
from .tune_regularization import (
    build_final_confirmation_candidates,
    run_dropout_tuning,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset_arcsinh.pkl")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output = Path("experiment_results/final_evaluation") / f"final_{stamp}.json"

    candidates = build_final_confirmation_candidates()
    print("Prepared locked final confirmation:")
    for candidate in candidates:
        print(f"- {candidate.name}: {candidate.architecture}")
    run_dropout_tuning(
        args.dataset,
        candidates,
        stage="final_confirmation",
        folds=5,
        epochs=FROZEN_EPOCHS,
        num_workers=args.workers,
        seeds=[0, 1, 2],
        device_name=args.device,
        output_path=output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
