#!/usr/bin/env python3
"""Move the first prediction epoch files from def-afyshe-ab to aip-afyshe.

This script scans resume-run prediction directories under the def root,
finds the earliest prediction epoch in each directory, and moves matching
npz files (predictions/targets/inputs) for that epoch to the mirrored
location under the aip root.

Default mode is dry-run. Use --execute to perform moves.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path


FILE_RE = re.compile(r"^(predictions|targets|inputs)_.+_epoch_(\d+)\.npz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move first-epoch prediction files from def-afyshe-ab runs "
            "to mirrored aip-afyshe directories."
        )
    )
    parser.add_argument(
        "--def-root",
        type=Path,
        default=Path("/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy"),
        help="Source repository root (def-afyshe-ab).",
    )
    parser.add_argument(
        "--aip-root",
        type=Path,
        default=Path("/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy"),
        help="Destination repository root (aip-afyshe).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files. If omitted, script runs in dry-run mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-directory details.",
    )
    parser.add_argument(
        "--include-non-resume",
        action="store_true",
        help=(
            "Also include non-resume prediction directories (e.g., under init_seed_*/predictions). "
            "Default behavior is resume-only."
        ),
    )
    return parser.parse_args()


def find_prediction_dirs(def_root: Path, include_non_resume: bool, verbose: bool) -> list[Path]:
    saved_models_root = def_root / "src" / "saved_models" # / "complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_preprocessed"
    if not saved_models_root.exists():
        print(f"WARNING: {saved_models_root} does not exist; nothing to scan.")
        return []

    
    print(f"Scanning for prediction directories under {saved_models_root}...")



    prediction_dirs: list[Path] = []
    seen_dirs = 0

    for root, dirnames, _ in os.walk(saved_models_root, topdown=True, followlinks=False):
        seen_dirs += 1
        if seen_dirs % 10000 == 0:
            print(f"  scanned {seen_dirs} directories...")

        root_path = Path(root)

        # Never descend into symlinked directories.
        dirnames[:] = [d for d in dirnames if not (root_path / d).is_symlink()]

        if root_path.name != "predictions":
            continue

        parent_name = root_path.parent.name
        is_resume_dir = parent_name.startswith("resume_from_epoch_") and "__freeze_" in parent_name

        if is_resume_dir or include_non_resume:
            if verbose:
                kind = "resume" if is_resume_dir else "non-resume"
                print(f"[FOUND {kind}] {root_path}")
            prediction_dirs.append(root_path)

        # predictions directories are leaves for our purpose.
        dirnames[:] = []

    print(f"Found {len(prediction_dirs)} prediction directories.")
    return prediction_dirs


def extract_epoch(filename: str) -> tuple[str, int] | None:
    match = FILE_RE.match(filename)
    if not match:
        return None
    kind, epoch = match.group(1), int(match.group(2))
    return kind, epoch


def earliest_prediction_epoch(pred_dir: Path) -> int | None:
    epochs: list[int] = []
    for path in pred_dir.glob("predictions_*_epoch_*.npz"):
        parsed = extract_epoch(path.name)
        if parsed is None:
            continue
        _, epoch = parsed
        epochs.append(epoch)
    if not epochs:
        return None
    return min(epochs)


def collect_files_for_epoch(pred_dir: Path, epoch: int) -> list[Path]:
    files: list[Path] = []
    for path in pred_dir.glob("*.npz"):
        parsed = extract_epoch(path.name)
        if parsed is None:
            continue
        _, file_epoch = parsed
        if file_epoch == epoch:
            files.append(path)
    return sorted(files)


def mapped_destination(src_path: Path, def_root: Path, aip_root: Path) -> Path:
    rel = src_path.relative_to(def_root)
    return aip_root / rel


def main() -> int:
    args = parse_args()

    # Keep user-provided roots stable (e.g. /home/rsaha/projects/...) without
    # resolving through symlink aliases like /project/....
    def_root = Path(os.path.abspath(os.path.expanduser(str(args.def_root))))
    aip_root = Path(os.path.abspath(os.path.expanduser(str(args.aip_root))))

    if not def_root.exists():
        print(f"ERROR: def root does not exist: {def_root}")
        return 2
    if not aip_root.exists():
        print(f"ERROR: aip root does not exist: {aip_root}")
        return 2

    print(f"Def root: {def_root}")
    pred_dirs = find_prediction_dirs(def_root, args.include_non_resume, args.verbose)
    if not pred_dirs:
        mode_label = "resume or non-resume" if args.include_non_resume else "resume"
        print(f"No {mode_label} prediction directories found under def root.")
        return 0

    moved = 0
    skipped_exists = 0
    skipped_empty = 0
    candidate_files = 0

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"Mode: {mode}")
    print(f"Found {len(pred_dirs)} prediction directories.")

    for pred_dir in sorted(pred_dirs):
        first_epoch = earliest_prediction_epoch(pred_dir)
        if first_epoch is None:
            skipped_empty += 1
            if args.verbose:
                print(f"[SKIP] No prediction epoch found in {pred_dir}")
            continue

        files = collect_files_for_epoch(pred_dir, first_epoch)
        if not files:
            skipped_empty += 1
            if args.verbose:
                print(f"[SKIP] No files found for earliest epoch {first_epoch} in {pred_dir}")
            continue

        if args.verbose:
            print(f"[DIR] {pred_dir} -> earliest epoch {first_epoch}, files={len(files)}")

        for src in files:
            candidate_files += 1
            dst = mapped_destination(src, def_root, aip_root)

            if dst.exists() and not args.overwrite:
                skipped_exists += 1
                print(f"[SKIP exists] {dst}")
                continue

            print(f"[MOVE] {src} -> {dst}")
            if args.execute:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() and args.overwrite:
                    dst.unlink()
                shutil.move(str(src), str(dst))
                moved += 1

    print("\nSummary")
    print(f"- candidate files: {candidate_files}")
    print(f"- moved files: {moved}")
    print(f"- skipped (destination exists): {skipped_exists}")
    print(f"- skipped (no usable epoch/files): {skipped_empty}")

    if not args.execute:
        print("Dry-run only. Re-run with --execute to apply moves.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
