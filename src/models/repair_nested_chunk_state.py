#!/usr/bin/env python3
"""Repair arbitrarily nested chunk_state resume directories.

This script detects resume run directories that were recursively created under
.../chunk_state/resume_from_epoch_* and promotes useful artifacts back to the
canonical top-level run root.

Default mode is dry-run. Use --execute to apply file moves/copies.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path


RESUME_DIR_RE = re.compile(r"^resume_from_epoch_\d+__freeze_.+")
CHKPT_RE = re.compile(r"^chunk_checkpoint_epoch_(\d+)\.pt$")


@dataclass
class MoveStat:
    moved: int = 0
    skipped_exists_same: int = 0
    skipped_exists_conflict: int = 0
    errors: int = 0


@dataclass
class RunReport:
    canonical_root: Path
    nested_roots: list[Path] = field(default_factory=list)
    checkpoint_epochs_top_before: list[int] = field(default_factory=list)
    checkpoint_epochs_nested: list[int] = field(default_factory=list)
    checkpoint_epochs_union_after: list[int] = field(default_factory=list)
    nested_prediction_files: int = 0
    skip_reason: str | None = None
    checkpoint_stat: MoveStat = field(default_factory=MoveStat)
    prediction_stat: MoveStat = field(default_factory=MoveStat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair recursively nested resume/chunk_state directories by promoting "
            "checkpoints and prediction files back to the canonical run root."
        )
    )
    parser.add_argument(
        "--saved-models-root",
        type=Path,
        default=Path("src/saved_models"),
        help="Root under which run directories are scanned.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply file operations. Default is dry-run.",
    )
    parser.add_argument(
        "--copy-instead-of-move",
        action="store_true",
        help="Copy files from nested trees instead of moving them.",
    )
    parser.add_argument(
        "--prune-nested",
        action="store_true",
        help=(
            "After successful promotion, delete nested resume directories that are "
            "under chunk_state. Ignored in dry-run mode."
        ),
    )
    parser.add_argument(
        "--min-idle-minutes",
        type=int,
        default=30,
        help=(
            "Skip runs with recent modification under their canonical root. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--only-canonical-contains",
        type=str,
        default="",
        help="Optional substring filter applied to canonical run root path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-file actions.",
    )
    parser.add_argument(
        "--prefer-larger-nested-predictions",
        action="store_true",
        help=(
            "For prediction files only: if destination exists and the nested source "
            "file is larger, replace destination with source."
        ),
    )
    return parser.parse_args()


def is_resume_dir_name(name: str) -> bool:
    return RESUME_DIR_RE.match(name) is not None


def collect_resume_dirs(saved_models_root: Path) -> list[Path]:
    out: list[Path] = []
    for root, dirnames, _ in os.walk(saved_models_root, topdown=True, followlinks=False):
        root_path = Path(root)
        print(f"Scanning {root_path}")
        dirnames[:] = [d for d in dirnames if not (root_path / d).is_symlink()]
        for d in dirnames:
            if is_resume_dir_name(d):
                out.append(root_path / d)
    return sorted(set(out))


def get_topmost_resume_ancestor(path: Path) -> Path | None:
    top: Path | None = None
    for part in [path, *path.parents]:
        if is_resume_dir_name(part.name):
            top = part
    return top


def is_nested_under_chunk_state(path: Path) -> bool:
    return "chunk_state" in path.parts and is_resume_dir_name(path.name)


def newest_mtime_under(path: Path) -> float:
    newest = 0.0
    for root, _, files in os.walk(path, topdown=True, followlinks=False):
        root_path = Path(root)
        try:
            newest = max(newest, root_path.stat().st_mtime)
        except OSError:
            pass
        for f in files:
            fp = root_path / f
            try:
                newest = max(newest, fp.stat().st_mtime)
            except OSError:
                continue
    return newest


def list_checkpoint_files(chunk_state_dir: Path) -> list[tuple[Path, int]]:
    files: list[tuple[Path, int]] = []
    if not chunk_state_dir.exists():
        return files
    for p in sorted(chunk_state_dir.glob("chunk_checkpoint_epoch_*.pt")):
        m = CHKPT_RE.match(p.name)
        if not m:
            continue
        files.append((p, int(m.group(1))))
    return files


def list_prediction_files(pred_dir: Path) -> list[Path]:
    if not pred_dir.exists():
        return []
    return sorted([p for p in pred_dir.glob("*.npz") if p.is_file()])


def safe_promote_file(
    src: Path,
    dst: Path,
    execute: bool,
    copy_instead_of_move: bool,
    verbose: bool,
    replace_if_src_larger: bool = False,
) -> tuple[str, str | None]:
    if dst.exists():
        try:
            src_size = src.stat().st_size
            dst_size = dst.stat().st_size
            if src_size == dst_size:
                return "skip_same", None
            if replace_if_src_larger and src_size > dst_size:
                if verbose:
                    op = "COPY" if copy_instead_of_move else "MOVE"
                    print(
                        f"[{op} REPLACE_LARGER] {src} ({src_size}) -> {dst} ({dst_size})"
                    )

                if not execute:
                    return "moved", None

                try:
                    if copy_instead_of_move:
                        shutil.copy2(src, dst)
                    else:
                        shutil.move(str(src), str(dst))
                    return "moved", None
                except OSError as exc:
                    return "error", f"File operation failed for {src} -> {dst}: {exc}"

            return "skip_conflict", f"Destination exists with different size: {dst}"
        except OSError as exc:
            return "error", f"Stat failed for {src} -> {dst}: {exc}"

    if verbose:
        op = "COPY" if copy_instead_of_move else "MOVE"
        print(f"[{op}] {src} -> {dst}")

    if not execute:
        return "moved", None

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if copy_instead_of_move:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))
        return "moved", None
    except OSError as exc:
        return "error", f"File operation failed for {src} -> {dst}: {exc}"


def update_stat(stat: MoveStat, status: str) -> None:
    if status == "moved":
        stat.moved += 1
    elif status == "skip_same":
        stat.skipped_exists_same += 1
    elif status == "skip_conflict":
        stat.skipped_exists_conflict += 1
    else:
        stat.errors += 1


def discover_run_groups(saved_models_root: Path) -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = {}
    for resume_dir in collect_resume_dirs(saved_models_root):
        print(f"Found resume dir: {resume_dir}")
        if not is_nested_under_chunk_state(resume_dir):
            continue
        canonical = get_topmost_resume_ancestor(resume_dir)
        if canonical is None:
            continue
        groups.setdefault(canonical, []).append(resume_dir)

    for canonical, nested in groups.items():
        groups[canonical] = sorted(set(nested))
    return groups


def repair_one_run(
    canonical_root: Path,
    nested_roots: list[Path],
    execute: bool,
    copy_instead_of_move: bool,
    prune_nested: bool,
    verbose: bool,
    prefer_larger_nested_predictions: bool,
) -> RunReport:
    report = RunReport(canonical_root=canonical_root, nested_roots=nested_roots)

    top_chunk_state = canonical_root / "chunk_state"
    top_predictions = canonical_root / "predictions"

    top_chkpts_before = list_checkpoint_files(top_chunk_state)
    report.checkpoint_epochs_top_before = sorted({e for _, e in top_chkpts_before})

    for nested in nested_roots:
        nested_chunk_state = nested / "chunk_state"
        nested_predictions = nested / "predictions"

        nested_chkpts = list_checkpoint_files(nested_chunk_state)
        report.checkpoint_epochs_nested.extend([e for _, e in nested_chkpts])

        for src, _ in nested_chkpts:
            dst = top_chunk_state / src.name
            status, err = safe_promote_file(
                src, dst, execute, copy_instead_of_move, verbose
            )
            update_stat(report.checkpoint_stat, status)
            if err and verbose:
                print(f"[WARN] {err}")

        pred_files = list_prediction_files(nested_predictions)
        report.nested_prediction_files += len(pred_files)
        for src in pred_files:
            dst = top_predictions / src.name
            status, err = safe_promote_file(
                src,
                dst,
                execute,
                copy_instead_of_move,
                verbose,
                replace_if_src_larger=prefer_larger_nested_predictions,
            )
            update_stat(report.prediction_stat, status)
            if err and verbose:
                print(f"[WARN] {err}")

    report.checkpoint_epochs_nested = sorted(set(report.checkpoint_epochs_nested))
    report.checkpoint_epochs_union_after = sorted(
        set(report.checkpoint_epochs_top_before).union(report.checkpoint_epochs_nested)
    )

    if execute and prune_nested and report.checkpoint_stat.errors == 0 and report.prediction_stat.errors == 0:
        for nested in sorted(nested_roots, key=lambda p: len(p.parts), reverse=True):
            try:
                shutil.rmtree(nested)
                if verbose:
                    print(f"[PRUNE] Removed nested tree: {nested}")
            except OSError as exc:
                print(f"[WARN] Failed to prune {nested}: {exc}")

    return report


def main() -> int:
    args = parse_args()

    saved_models_root = Path(os.path.abspath(os.path.expanduser(str(args.saved_models_root))))
    if not saved_models_root.exists():
        print(f"ERROR: saved-models root does not exist: {saved_models_root}")
        return 2

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"Mode: {mode}")
    print(f"Saved-models root: {saved_models_root}")

    groups = discover_run_groups(saved_models_root)
    if args.only_canonical_contains:
        groups = {
            k: v
            for k, v in groups.items()
            if args.only_canonical_contains in str(k)
        }

    print(f"Found {len(groups)} canonical runs with nested resume trees under chunk_state.")
    if not groups:
        return 0

    now = __import__("time").time()
    reports: list[RunReport] = []

    for canonical_root, nested_roots in sorted(groups.items()):
        report = RunReport(canonical_root=canonical_root, nested_roots=nested_roots)

        if args.min_idle_minutes > 0:
            newest_mtime = newest_mtime_under(canonical_root)
            idle_minutes = (now - newest_mtime) / 60.0 if newest_mtime > 0 else 1e9
            if idle_minutes < args.min_idle_minutes:
                report.skip_reason = (
                    f"active-recently (idle={idle_minutes:.1f}m < {args.min_idle_minutes}m)"
                )
                reports.append(report)
                continue

        repaired = repair_one_run(
            canonical_root=canonical_root,
            nested_roots=nested_roots,
            execute=args.execute,
            copy_instead_of_move=args.copy_instead_of_move,
            prune_nested=args.prune_nested,
            verbose=args.verbose,
            prefer_larger_nested_predictions=args.prefer_larger_nested_predictions,
        )
        reports.append(repaired)

    print("\nPer-run summary")
    for r in reports:
        nested_count = len(r.nested_roots)
        if r.skip_reason:
            print(f"- {r.canonical_root}")
            print(f"  skipped: {r.skip_reason}; nested_roots={nested_count}")
            continue

        top_max = max(r.checkpoint_epochs_top_before) if r.checkpoint_epochs_top_before else None
        nested_max = max(r.checkpoint_epochs_nested) if r.checkpoint_epochs_nested else None
        union_max = max(r.checkpoint_epochs_union_after) if r.checkpoint_epochs_union_after else None

        print(f"- {r.canonical_root}")
        print(
            "  nested_roots={} chkpt(top_max={}, nested_max={}, union_max={}) ".format(
                nested_count,
                top_max,
                nested_max,
                union_max,
            )
            + "pred(nested_files={})".format(r.nested_prediction_files)
        )
        print(
            "  checkpoint: moved={} skip_same={} conflicts={} errors={}".format(
                r.checkpoint_stat.moved,
                r.checkpoint_stat.skipped_exists_same,
                r.checkpoint_stat.skipped_exists_conflict,
                r.checkpoint_stat.errors,
            )
        )
        print(
            "  predictions: moved={} skip_same={} conflicts={} errors={}".format(
                r.prediction_stat.moved,
                r.prediction_stat.skipped_exists_same,
                r.prediction_stat.skipped_exists_conflict,
                r.prediction_stat.errors,
            )
        )

    eligible = [r for r in reports if not r.skip_reason]
    total_chkpt = MoveStat()
    total_pred = MoveStat()
    for r in eligible:
        total_chkpt.moved += r.checkpoint_stat.moved
        total_chkpt.skipped_exists_same += r.checkpoint_stat.skipped_exists_same
        total_chkpt.skipped_exists_conflict += r.checkpoint_stat.skipped_exists_conflict
        total_chkpt.errors += r.checkpoint_stat.errors

        total_pred.moved += r.prediction_stat.moved
        total_pred.skipped_exists_same += r.prediction_stat.skipped_exists_same
        total_pred.skipped_exists_conflict += r.prediction_stat.skipped_exists_conflict
        total_pred.errors += r.prediction_stat.errors

    print("\nTotals")
    print(f"- runs scanned: {len(reports)}")
    print(f"- runs eligible: {len(eligible)}")
    print(f"- runs skipped: {len(reports) - len(eligible)}")
    print(
        "- checkpoints: moved={} skip_same={} conflicts={} errors={}".format(
            total_chkpt.moved,
            total_chkpt.skipped_exists_same,
            total_chkpt.skipped_exists_conflict,
            total_chkpt.errors,
        )
    )
    print(
        "- predictions: moved={} skip_same={} conflicts={} errors={}".format(
            total_pred.moved,
            total_pred.skipped_exists_same,
            total_pred.skipped_exists_conflict,
            total_pred.errors,
        )
    )

    if not args.execute:
        print("\nDry-run only. Re-run with --execute to apply changes.")
    elif args.prune_nested:
        print("\nPruning requested: nested resume trees were removed for successful runs.")
    else:
        print("\nNested trees were not pruned. Re-run with --prune-nested if desired.")

    if total_chkpt.errors > 0 or total_pred.errors > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
