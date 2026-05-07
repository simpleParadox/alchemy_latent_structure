#!/usr/bin/env python3
"""
check_stale_pickles.py

Batch-check frozen-layer pickle files for completeness. A pickle is "stale"
if it has fewer metric values than expected AND the run did NOT converge
(i.e., it was not legitimately early-stopped).

Expected values = 999 - freeze_epoch  (predictions from freeze_epoch+1 to 999)

Early-stopping detection:
  If the last `patience` values of the convergence metric are all >= threshold,
  the run is considered converged and NOT stale — even if it has fewer epochs
  than expected. This matches the logic in find_incomplete_runs.py.

The freeze_epoch is parsed from the filename pattern:
    ..._freeze_epoch_{N}_...

Usage:
    python src/models/check_stale_pickles.py <pickle_dir>
    python src/models/check_stale_pickles.py <pickle_dir> --threshold 0.975 --patience 10

Prints one stale pickle path per line to stdout.
Prints summary info to stderr so stdout can be captured cleanly.
"""
import argparse
import os
import re
import sys
import glob
import pickle


# Defaults matching find_incomplete_runs.py
DEFAULT_METRIC_KEY = "predicted_exact_out_of_all_108"
DEFAULT_THRESHOLD = 0.975
DEFAULT_PATIENCE = 10


def parse_freeze_epoch(filename: str) -> int | None:
    """Extract freeze_epoch from filename like '..._freeze_epoch_17_...'."""
    match = re.search(r"freeze_epoch_(\d+)", filename)
    return int(match.group(1)) if match else None


def check_early_stopped(
    seed_results: dict,
    metric_key: str,
    threshold: float,
    patience: int,
) -> bool:
    """Check if a pickle's run converged (early-stopped legitimately).

    Returns True if the last `patience` values of `metric_key` are all
    >= `threshold` for at least one seed.
    """
    for seed_key, metrics in seed_results.items():
        values = metrics.get(metric_key)
        if values is None or not hasattr(values, "__len__"):
            continue
        if len(values) < patience:
            continue
        tail = values[-patience:]
        if all(v >= threshold for v in tail):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Check frozen-layer pickles for staleness."
    )
    parser.add_argument("pickle_dir", help="Directory containing pickle files.")
    parser.add_argument(
        "--metric_key",
        default=DEFAULT_METRIC_KEY,
        help=f"Metric key to check for convergence (default: {DEFAULT_METRIC_KEY}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Convergence threshold (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Number of consecutive epochs above threshold to consider converged (default: {DEFAULT_PATIENCE}).",
    )
    args = parser.parse_args()

    pickle_dir = args.pickle_dir
    if not os.path.isdir(pickle_dir):
        print(f"Directory not found: {pickle_dir}", file=sys.stderr)
        sys.exit(1)

    # Only check frozen-layer pickles (contain "freeze_epoch" in filename)
    pkl_files = sorted(glob.glob(os.path.join(pickle_dir, "*freeze_epoch*.pkl")))
    total = 0
    stale = 0
    complete = 0
    early_stopped = 0

    for pkl_path in pkl_files:
        filename = os.path.basename(pkl_path)
        freeze_epoch = parse_freeze_epoch(filename)

        if freeze_epoch is None:
            print(f"  Warning: Could not parse freeze_epoch from {filename}", file=sys.stderr)
            continue

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"  Error loading {pkl_path}: {e}", file=sys.stderr)
            print(pkl_path)  # treat unloadable pickles as stale
            stale += 1
            total += 1
            continue

        total += 1
        expected_values = 999 - freeze_epoch  # predictions from freeze_epoch+1 to 999
        seed_results = data.get("seed_results", {})

        # Check the first available seed/metric for length
        actual_values = None
        for seed_key, metrics in seed_results.items():
            for metric_key, values in metrics.items():
                if hasattr(values, "__len__"):
                    actual_values = len(values)
                    break
            break  # only need to check one seed/metric

        if actual_values is None:
            print(f"  Warning: No metric values found in {filename}", file=sys.stderr)
            print(pkl_path)  # treat as stale
            stale += 1
            continue

        if actual_values < expected_values:
            # Pickle is shorter than expected — check if it converged (early-stopped)
            converged = check_early_stopped(
                seed_results,
                metric_key=args.metric_key,
                threshold=args.threshold,
                patience=args.patience,
            )
            if converged:
                early_stopped += 1
                print(
                    f"  Early-stopped (not stale): {filename} — "
                    f"{actual_values}/{expected_values} values, "
                    f"tail {args.patience} >= {args.threshold}",
                    file=sys.stderr,
                )
            else:
                print(pkl_path)  # stdout: stale path
                stale += 1
                print(
                    f"  Incomplete: {filename} — {actual_values}/{expected_values} values",
                    file=sys.stderr,
                )
        else:
            complete += 1

    print(
        f"\nChecked {total} frozen pickles: "
        f"{complete} complete, {early_stopped} early-stopped, {stale} stale",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
