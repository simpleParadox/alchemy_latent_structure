#!/usr/bin/env python3
"""
sync_prediction_dirs_from_list.py

Sync remote prediction directories listed in a text file to a local prefix,
while preserving the path suffix starting at an anchor (default: dm_alchemy).

Typical mapping:
  remote: /home/rsaha/scratch/dm_alchemy/src/saved_models/.../predictions
  local:  /home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/.../predictions

Features:
- Dry-run mode (default behavior if --dry_run is passed)
- Automatic SSH ControlMaster session setup/teardown
- Optional explicit SSH options
- Summary table before transfer
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

PRED_PATTERN = "predictions_classification_epoch_*.npz"
CHECKPOINT_PATTERNS = [
    "chunk_checkpoint_epoch_*.pt",
    "best_model_epoch_*.pt",
    "run_state.json",
    "wandb_run_id.txt",
]


@dataclass
class PathPlan:
    source_dir: str
    target_dir: str
    source_is_remote: bool
    local_source_exists: bool
    local_source_npz_count: int
    local_target_exists: bool
    local_target_npz_count: int


@dataclass
class ConflictRow:
    source_dir: str
    target_dir: str
    local_npz_count: int
    remote_npz_count: Optional[int]
    reason: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sync remote prediction directories from a file into a local prefix."
    )
    p.add_argument(
        "--direction",
        type=str,
        choices=["pull", "push"],
        default="pull",
        help=(
            "Transfer direction: pull (remote->local, default) or push (local->remote)."
        ),
    )
    p.add_argument("--remote_host", type=str, required=True, help="Remote SSH host.")
    p.add_argument("--paths_file", type=str, required=True, help="Text file with remote prediction directories.")
    p.add_argument(
        "--local_prefix",
        type=str,
        default="",
        help="Local prefix replacing everything before anchor in pull mode.",
    )
    p.add_argument(
        "--remote_prefix",
        type=str,
        default="",
        help="Remote prefix replacing everything before anchor in push mode.",
    )
    p.add_argument(
        "--anchor",
        type=str,
        default="dm_alchemy",
        help="Path anchor retained in mapped destination (default: dm_alchemy).",
    )
    p.add_argument(
        "--local_suffix",
        type=str,
        default="",
        help=(
            "Optional subdirectory inserted in destination path "
            "(e.g., sync_2026_04_15)."
        ),
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print plan and commands without transferring files.",
    )
    p.add_argument(
        "--allow_existing_without_suffix",
        action="store_true",
        help=(
            "Allow transfer to proceed when local_suffix is empty and mapped local directories already exist."
        ),
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip transfer confirmation prompt.",
    )
    p.add_argument(
        "--ssh_option",
        action="append",
        default=[],
        help="Extra ssh option, can be repeated. Example: --ssh_option='-o ConnectTimeout=20'",
    )
    p.add_argument(
        "--no_auto_control_master",
        action="store_true",
        help="Disable automatic ControlMaster setup.",
    )
    p.add_argument(
        "--ssh_timeout_seconds",
        type=int,
        default=120,
        help="Timeout for SSH operations in seconds (default: 120).",
    )
    p.add_argument(
        "--artifact_type",
        type=str,
        choices=["predictions", "checkpoints", "both"],
        default="predictions",
        help=(
            "What to transfer: prediction files, checkpoint files, or both. "
            "Default is predictions."
        ),
    )
    p.add_argument(
        "--include_pattern",
        type=str,
        action="append",
        default=None,
        help=(
            "Glob pattern to transfer with rsync; can be repeated. "
            "If omitted, defaults are chosen from --artifact_type."
        ),
    )
    return p.parse_args()


def effective_patterns(args: argparse.Namespace) -> List[str]:
    if args.include_pattern:
        return args.include_pattern
    if args.artifact_type == "predictions":
        return [PRED_PATTERN]
    if args.artifact_type == "checkpoints":
        return list(CHECKPOINT_PATTERNS)
    return [PRED_PATTERN, *CHECKPOINT_PATTERNS]


def flatten_ssh_options(values: List[str]) -> List[str]:
    flat: List[str] = []
    for item in values:
        flat.extend(shlex.split(item))
    return flat


def has_ssh_option(ssh_options: List[str], key: str) -> bool:
    prefix = f"{key}="
    for i, token in enumerate(ssh_options):
        if token == "-o" and i + 1 < len(ssh_options):
            if ssh_options[i + 1].startswith(prefix):
                return True
        if token.startswith("-o") and token[2:].startswith(prefix):
            return True
    return False


def extract_control_path(ssh_options: List[str]) -> Optional[str]:
    for i, token in enumerate(ssh_options):
        if token == "-o" and i + 1 < len(ssh_options):
            opt = ssh_options[i + 1]
            if opt.startswith("ControlPath="):
                return os.path.expanduser(opt.split("=", 1)[1])
        if token.startswith("-oControlPath="):
            return os.path.expanduser(token.split("=", 1)[1])
    return None


def setup_control_master(args: argparse.Namespace) -> Tuple[List[str], Optional[str], bool]:
    ssh_options = flatten_ssh_options(args.ssh_option)
    if args.no_auto_control_master:
        return ssh_options, extract_control_path(ssh_options), False

    opts = list(ssh_options)
    if not has_ssh_option(opts, "ControlPath"):
        cp = os.path.expanduser(f"~/.ssh/cm-sync-{args.remote_host.replace('/', '_')}-{os.getpid()}")
        opts.extend(["-o", f"ControlPath={cp}"])
    if not has_ssh_option(opts, "ControlMaster"):
        opts.extend(["-o", "ControlMaster=auto"])
    if not has_ssh_option(opts, "ControlPersist"):
        opts.extend(["-o", "ControlPersist=15m"])

    control_path = extract_control_path(opts)
    cmd = ["ssh", *opts, "-MNf", args.remote_host]
    try:
        subprocess.run(cmd, check=True, timeout=args.ssh_timeout_seconds)
        print("Established SSH ControlMaster session.")
        return opts, control_path, True
    except subprocess.SubprocessError:
        print("WARNING: Could not establish ControlMaster automatically; falling back to direct SSH.")
        return ssh_options, extract_control_path(ssh_options), False


def teardown_control_master(remote_host: str, control_path: Optional[str], started: bool) -> None:
    if not started or not control_path:
        return
    cmd = ["ssh", "-S", control_path, "-O", "exit", remote_host]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_remote_paths(paths_file: str) -> List[str]:
    rows: List[str] = []
    with open(paths_file, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.rstrip("/"))
    return rows


def map_path_to_prefix(path: str, target_prefix: str, anchor: str, local_suffix: str) -> str:
    anchor_token = f"/{anchor}/"
    i = path.find(anchor_token)
    if i < 0:
        raise ValueError(f"Anchor '/{anchor}/' not found in path: {path}")
    suffix = path[i + 1 :]

    # Insert local_suffix right after the init_seed_* directory so the
    # run hierarchy stays intact while avoiding overwrites.
    if local_suffix:
        parts = suffix.split("/")
        try:
            init_idx = next(idx for idx, part in enumerate(parts) if part.startswith("init_seed_"))
        except StopIteration as exc:
            raise ValueError(
                f"Could not find an 'init_seed_*' segment in path suffix: {suffix}"
            ) from exc
        parts.insert(init_idx + 1, local_suffix.strip("/"))
        suffix = "/".join(parts)

    base = target_prefix.rstrip("/")
    return os.path.join(base, suffix)


def expand_source_dirs(paths: List[str], artifact_type: str) -> List[str]:
    expanded: List[str] = []
    for p in paths:
        norm = p.rstrip("/")
        base = os.path.basename(norm)
        parent = os.path.dirname(norm)

        if artifact_type in {"predictions", "both"}:
            if base == "chunk_state":
                expanded.append(os.path.join(parent, "predictions"))
            elif base == "predictions":
                expanded.append(norm)
            elif "resume_from_epoch_" in norm:
                expanded.append(os.path.join(norm, "predictions"))

        if artifact_type in {"checkpoints", "both"}:
            if base == "predictions":
                expanded.append(os.path.join(parent, "chunk_state"))
            elif base == "chunk_state":
                expanded.append(norm)
            elif "resume_from_epoch_" in norm:
                expanded.append(os.path.join(norm, "chunk_state"))

    deduped: List[str] = []
    seen = set()
    for p in expanded:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def local_match_count(path: str, include_patterns: List[str]) -> int:
    if not os.path.isdir(path):
        return 0

    if not include_patterns:
        return 0

    or_expr = " -o ".join([f"-name {shlex.quote(p)}" for p in include_patterns])
    find_expr = f"find {shlex.quote(path)} -maxdepth 1 -type f \\( {or_expr} \\) | wc -l"
    cmd = ["bash", "-lc", find_expr]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return 0
    try:
        return int(proc.stdout.strip() or "0")
    except ValueError:
        return 0


def build_pull_plan(
    source_remote_dirs: List[str],
    local_prefix: str,
    anchor: str,
    include_patterns: List[str],
    local_suffix: str,
) -> List[PathPlan]:
    plan: List[PathPlan] = []
    for remote_dir in source_remote_dirs:
        local_dir = map_path_to_prefix(remote_dir, local_prefix, anchor, local_suffix)
        exists = os.path.isdir(local_dir)
        count = local_match_count(local_dir, include_patterns)
        plan.append(
            PathPlan(
                source_dir=remote_dir,
                target_dir=local_dir,
                source_is_remote=True,
                local_source_exists=True,
                local_source_npz_count=0,
                local_target_exists=exists,
                local_target_npz_count=count,
            )
        )
    return plan


def build_push_plan(
    source_local_dirs: List[str],
    remote_prefix: str,
    anchor: str,
    include_patterns: List[str],
    local_suffix: str,
) -> List[PathPlan]:
    plan: List[PathPlan] = []
    for local_dir in source_local_dirs:
        remote_dir = map_path_to_prefix(local_dir, remote_prefix, anchor, local_suffix)
        source_exists = os.path.isdir(local_dir)
        source_count = local_match_count(local_dir, include_patterns)
        plan.append(
            PathPlan(
                source_dir=local_dir,
                target_dir=remote_dir,
                source_is_remote=False,
                local_source_exists=source_exists,
                local_source_npz_count=source_count,
                local_target_exists=False,
                local_target_npz_count=0,
            )
        )
    return plan


def print_plan(plan: List[PathPlan], direction: str) -> None:
    if not plan:
        print("No paths found in input file.")
        return

    if direction == "pull":
        fmt = "{:<4} {:<7} {:<10} {:<90}"
        print(fmt.format("#", "exists", "local_npz", "local_dir"))
        print("-" * 130)
        for i, row in enumerate(plan, start=1):
            print(fmt.format(i, str(row.local_target_exists), row.local_target_npz_count, row.target_dir))
    else:
        fmt = "{:<4} {:<7} {:<10} {:<90}"
        print(fmt.format("#", "exists", "local_npz", "local_source_dir"))
        print("-" * 130)
        for i, row in enumerate(plan, start=1):
            print(fmt.format(i, str(row.local_source_exists), row.local_source_npz_count, row.source_dir))

    print("\nSource -> Target mapping:")
    # for row in plan:
        # print(f"SOURCE: {row.source_dir}")
        # print(f"TARGET: {row.target_dir}")
        # print()


def print_skipped_missing_sources(rows: List[PathPlan]) -> None:
    if not rows:
        return
    print(f"\nSkipping {len(rows)} rows because source directories do not exist locally:")
    for row in rows:
        print(f"  - {row.source_dir}")


def confirm_transfer(args: argparse.Namespace, n: int) -> bool:
    if args.dry_run:
        return False
    if args.yes:
        return True

    print(f"About to transfer {n} prediction directories. Type YES to continue: ", end="", flush=True)
    ans = input().strip()
    return ans == "YES"


def rsync_one(
    direction: str,
    remote_host: str,
    ssh_options: List[str],
    source_dir: str,
    target_dir: str,
    include_patterns: List[str],
    timeout_seconds: int,
) -> bool:
    if direction == "pull":
        os.makedirs(target_dir, exist_ok=True)
    else:
        if not os.path.isdir(source_dir):
            return False
        mkdir_cmd = ["ssh", *ssh_options, remote_host, f"mkdir -p {shlex.quote(target_dir)}"]
        try:
            mkdir_proc = subprocess.run(mkdir_cmd, timeout=timeout_seconds)
            if mkdir_proc.returncode != 0:
                return False
        except subprocess.TimeoutExpired:
            return False

    ssh_inner = "ssh " + " ".join(shlex.quote(opt) for opt in ssh_options)
    if direction == "pull":
        src = f"{remote_host}:{source_dir.rstrip('/')}/"
        dst = f"{target_dir.rstrip('/')}/"
    else:
        src = f"{source_dir.rstrip('/')}/"
        dst = f"{remote_host}:{target_dir.rstrip('/')}/"

    cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e",
        ssh_inner,
        "--include",
        "*/",
    ]

    for pattern in include_patterns:
        cmd.extend(["--include", pattern])

    cmd.extend([
        "--exclude",
        "*",
        src,
        dst,
    ])
    try:
        proc = subprocess.run(cmd, timeout=timeout_seconds)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def remote_npz_count(
    remote_host: str,
    ssh_options: List[str],
    remote_dir: str,
    include_patterns: List[str],
    timeout_seconds: int,
) -> Optional[int]:
    if not include_patterns:
        return 0

    quoted_dir = shlex.quote(remote_dir)
    or_expr = " -o ".join([f"-name {shlex.quote(p)}" for p in include_patterns])
    cmd = (
        "if [ -d {d} ]; then "
        "find {d} -maxdepth 1 -type f \\\(" + or_expr + "\\\) | wc -l; "
        "else echo __MISSING__; fi"
    ).format(d=quoted_dir)

    ssh_cmd = ["ssh", *ssh_options, remote_host, cmd]
    try:
        proc = subprocess.run(
            ssh_cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return None

    if proc.returncode != 0:
        return None

    out = (proc.stdout or "").strip()
    if "__MISSING__" in out:
        return 0

    try:
        return int(out.splitlines()[-1].strip())
    except Exception:
        return None


def compute_pull_conflicts(
    plan: List[PathPlan],
    remote_host: str,
    ssh_options: List[str],
    include_patterns: List[str],
    timeout_seconds: int,
) -> List[ConflictRow]:
    conflicts: List[ConflictRow] = []
    for row in plan:
        if not row.local_target_exists:
            continue
        rcount = remote_npz_count(
            remote_host=remote_host,
            ssh_options=ssh_options,
            remote_dir=row.source_dir,
            include_patterns=include_patterns,
            timeout_seconds=timeout_seconds,
        )
        conflicts.append(
            ConflictRow(
                source_dir=row.source_dir,
                target_dir=row.target_dir,
                local_npz_count=row.local_target_npz_count,
                remote_npz_count=rcount,
                reason="existing_local_target",
            )
        )
    return conflicts


def compute_push_conflicts(
    plan: List[PathPlan],
    remote_host: str,
    ssh_options: List[str],
    include_patterns: List[str],
    timeout_seconds: int,
) -> List[ConflictRow]:
    conflicts: List[ConflictRow] = []
    for row in plan:
        if not row.local_source_exists:
            continue
        local_count = row.local_source_npz_count
        rcount = remote_npz_count(
            remote_host=remote_host,
            ssh_options=ssh_options,
            remote_dir=row.target_dir,
            include_patterns=include_patterns,
            timeout_seconds=timeout_seconds,
        )
        # In push mode, skip only when remote has at least as many prediction files.
        if rcount is not None and rcount >= local_count:
            conflicts.append(
                ConflictRow(
                    source_dir=row.source_dir,
                    target_dir=row.target_dir,
                    local_npz_count=local_count,
                    remote_npz_count=rcount,
                    reason="remote_has_gte_local",
                )
            )
    return conflicts


def print_conflicts(conflicts: List[ConflictRow], direction: str) -> None:
    if not conflicts:
        return

    fmt = "{:<4} {:<10} {:<10} {:<10} {:<90}"
    if direction == "pull":
        print("\nConflicts (existing local target directories with empty local_suffix):")
    else:
        print("\nConflicts (remote target has >= local npz count with empty local_suffix):")
    print(fmt.format("#", "local_npz", "remote_npz", "delta", "target_dir"))
    print("-" * 130)
    for i, row in enumerate(conflicts, start=1):
        if row.remote_npz_count is None:
            remote_txt = "unknown"
            delta_txt = "unknown"
        else:
            remote_txt = str(row.remote_npz_count)
            delta_txt = str(row.remote_npz_count - row.local_npz_count)
        print(fmt.format(i, row.local_npz_count, remote_txt, delta_txt, row.target_dir))

    print("\nConflict details:")
    for row in conflicts:
        print(f"SOURCE: {row.source_dir}")
        print(f"TARGET: {row.target_dir}")
        print(f"REASON: {row.reason}")
        print()


def main() -> None:
    args = parse_args()
    include_patterns = effective_patterns(args)

    if args.direction == "pull" and not args.local_prefix:
        raise ValueError("--local_prefix is required when --direction pull")
    if args.direction == "push" and not args.remote_prefix:
        raise ValueError("--remote_prefix is required when --direction push")

    listed_paths = read_remote_paths(args.paths_file)
    expanded_paths = expand_source_dirs(listed_paths, args.artifact_type)
    if not expanded_paths:
        print("No transferable directories derived from input paths and artifact_type.")
        return

    if args.direction == "pull":
        plan = build_pull_plan(
            expanded_paths,
            args.local_prefix,
            args.anchor,
            include_patterns,
            args.local_suffix,
        )
    else:
        plan = build_push_plan(
            expanded_paths,
            args.remote_prefix,
            args.anchor,
            include_patterns,
            args.local_suffix,
        )
    print_plan(plan, args.direction)

    ssh_options: List[str] = []
    control_path: Optional[str] = None
    started = False

    ok = 0
    fail = 0
    transfer_plan: List[PathPlan] = list(plan)
    skipped_existing: List[PathPlan] = []
    skipped_missing_source: List[PathPlan] = []
    try:
        if args.direction == "push":
            skipped_missing_source = [row for row in transfer_plan if not row.local_source_exists]
            transfer_plan = [row for row in transfer_plan if row.local_source_exists]
            print_skipped_missing_sources(skipped_missing_source)

        if not args.local_suffix:
            ssh_options, control_path, started = setup_control_master(args)
            if args.direction == "pull":
                conflicts = compute_pull_conflicts(
                    plan=transfer_plan,
                    remote_host=args.remote_host,
                    ssh_options=ssh_options,
                    include_patterns=include_patterns,
                    timeout_seconds=args.ssh_timeout_seconds,
                )
            else:
                conflicts = compute_push_conflicts(
                    plan=transfer_plan,
                    remote_host=args.remote_host,
                    ssh_options=ssh_options,
                    include_patterns=include_patterns,
                    timeout_seconds=args.ssh_timeout_seconds,
                )
            print_conflicts(conflicts, args.direction)
            if conflicts and not args.allow_existing_without_suffix:
                conflict_targets = {c.target_dir for c in conflicts}
                skipped_existing = [row for row in transfer_plan if row.target_dir in conflict_targets]
                transfer_plan = [row for row in transfer_plan if row.target_dir not in conflict_targets]
                print("Conflict protection is active (local_suffix is empty).")
                if args.direction == "pull":
                    print("Existing local target directories will be skipped.")
                else:
                    print("Remote targets with >= local npz count will be skipped.")
                if args.direction == "pull":
                    print(
                        "Use --allow_existing_without_suffix if you intentionally want to transfer into existing local target directories."
                    )
                else:
                    print(
                        "Use --allow_existing_without_suffix if you intentionally want to transfer into remote targets even when they already have >= local npz count."
                    )

        if args.dry_run:
            if skipped_existing:
                if args.direction == "pull":
                    print(f"\n[DRY RUN] Skipping {len(skipped_existing)} existing local target directories:")
                else:
                    print(
                        f"\n[DRY RUN] Skipping {len(skipped_existing)} remote targets with >= local npz count:"
                    )
                for row in skipped_existing:
                    print(f"  - {row.target_dir}")
            if skipped_missing_source:
                print(f"[DRY RUN] Skipping {len(skipped_missing_source)} rows with missing local source dirs.")
            print(f"[DRY RUN] Would transfer {len(transfer_plan)} directories.")
            print("\n[DRY RUN] No files transferred.")
            return

        if skipped_existing:
            if args.direction == "pull":
                print(f"\nSkipping {len(skipped_existing)} existing local target directories:")
            else:
                print(f"\nSkipping {len(skipped_existing)} remote targets with >= local npz count:")
            for row in skipped_existing:
                print(f"  - {row.target_dir}")

        if skipped_missing_source:
            print(f"\nSkipped {len(skipped_missing_source)} rows with missing local source directories.")

        if not transfer_plan:
            print("No directories to transfer after applying conflict protection.")
            return

        if not confirm_transfer(args, len(transfer_plan)):
            print("Transfer cancelled.")
            return

        if not ssh_options:
            ssh_options, control_path, started = setup_control_master(args)

        for row in transfer_plan:
            print("\n" + "=" * 80)
            print(f"Transferring: {row.source_dir}")
            print(f"To:          {row.target_dir}")
            print("=" * 80)
            success = rsync_one(
                direction=args.direction,
                remote_host=args.remote_host,
                ssh_options=ssh_options,
                source_dir=row.source_dir,
                target_dir=row.target_dir,
                include_patterns=include_patterns,
                timeout_seconds=args.ssh_timeout_seconds,
            )
            if success:
                ok += 1
            else:
                fail += 1
                print("FAILED transfer for:", row.source_dir)
    finally:
        teardown_control_master(args.remote_host, control_path, started)

    print("\nTransfer summary")
    print(f"  Success: {ok}")
    print(f"  Failed:  {fail}")
    if skipped_existing:
        print(f"  Skipped existing: {len(skipped_existing)}")
    if skipped_missing_source:
        print(f"  Skipped missing source: {len(skipped_missing_source)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
