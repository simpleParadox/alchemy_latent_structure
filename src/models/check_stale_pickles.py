#!/usr/bin/env python3
"""
check_stale_pickles.py

Batch-check frozen-layer pickle files for completeness. A pickle is "stale"
if it has fewer metric values than expected (i.e., the training run that
generated it was incomplete).

Expected values = 999 - freeze_epoch  (predictions from epoch freeze_epoch+1 to 999)

The freeze_epoch is parsed from the filename pattern:
    ..._freeze_epoch_{N}_...

Usage:
    python src/models/check_stale_pickles.py <pickle_dir>

Prints one stale pickle path per line to stdout.
Prints summary info to stderr so stdout can be captured cleanly.
"""
import os
import re
import sys
import glob
import pickle


def parse_freeze_epoch(filename: str) -> int | None:
    """Extract freeze_epoch from filename like '..._freeze_epoch_17_...'."""
    match = re.search(r"freeze_epoch_(\d+)", filename)
    return int(match.group(1)) if match else None


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_stale_pickles.py <pickle_dir>", file=sys.stderr)
        sys.exit(1)

    pickle_dir = sys.argv[1]
    if not os.path.isdir(pickle_dir):
        print(f"Directory not found: {pickle_dir}", file=sys.stderr)
        sys.exit(1)

    # Only check frozen-layer pickles (contain "freeze_epoch" in filename)
    pkl_files = sorted(glob.glob(os.path.join(pickle_dir, "*freeze_epoch*.pkl")))
    total = 0
    stale = 0
    complete = 0

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
            print(pkl_path)  # stdout: stale path
            stale += 1
            print(
                f"  Incomplete: {filename} — {actual_values}/{expected_values} values",
                file=sys.stderr,
            )
        else:
            complete += 1

    print(f"\nChecked {total} frozen pickles: {complete} complete, {stale} stale", file=sys.stderr)


if __name__ == "__main__":
    main()
