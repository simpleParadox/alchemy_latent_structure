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


@dataclass
class PathPlan:
    remote_dir: str
    local_dir: str
    local_exists: bool
    local_npz_count: int


@dataclass
class ConflictRow:
    remote_dir: str
    local_dir: str
    local_npz_count: int
    remote_npz_count: Optional[int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sync remote prediction directories from a file into a local prefix."
    )
    p.add_argument("--remote_host", type=str, required=True, help="Remote SSH host.")
    p.add_argument("--paths_file", type=str, required=True, help="Text file with remote prediction directories.")
    p.add_argument(
        "--local_prefix",
        type=str,
        required=True,
        help="Local prefix to replace everything before the anchor.",
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
            "Optional subdirectory under local_prefix to store all synced paths "
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
        "--include_pattern",
        type=str,
        default=PRED_PATTERN,
        help=f"Glob pattern to transfer with rsync (default: {PRED_PATTERN}).",
    )
    return p.parse_args()


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


def map_remote_to_local(remote_dir: str, local_prefix: str, anchor: str, local_suffix: str) -> str:
    anchor_token = f"/{anchor}/"
    i = remote_dir.find(anchor_token)
    if i < 0:
        raise ValueError(f"Anchor '/{anchor}/' not found in remote path: {remote_dir}")
    suffix = remote_dir[i + 1 :]

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

    base = local_prefix.rstrip("/")
    return os.path.join(base, suffix)


def local_npz_count(path: str, include_pattern: str) -> int:
    if not os.path.isdir(path):
        return 0
    # Use shell glob count via find for consistency with pattern semantics.
    cmd = ["bash", "-lc", f"find {shlex.quote(path)} -maxdepth 1 -type f -name {shlex.quote(include_pattern)} | wc -l"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return 0
    try:
        return int(proc.stdout.strip() or "0")
    except ValueError:
        return 0


def build_plan(
    remote_dirs: List[str],
    local_prefix: str,
    anchor: str,
    include_pattern: str,
    local_suffix: str,
) -> List[PathPlan]:
    plan: List[PathPlan] = []
    for remote_dir in remote_dirs:
        local_dir = map_remote_to_local(remote_dir, local_prefix, anchor, local_suffix)
        exists = os.path.isdir(local_dir)
        count = local_npz_count(local_dir, include_pattern)
        plan.append(
            PathPlan(
                remote_dir=remote_dir,
                local_dir=local_dir,
                local_exists=exists,
                local_npz_count=count,
            )
        )
    return plan


def print_plan(plan: List[PathPlan]) -> None:
    if not plan:
        print("No paths found in input file.")
        return

    fmt = "{:<4} {:<7} {:<10} {:<90}"
    print(fmt.format("#", "exists", "local_npz", "local_dir"))
    print("-" * 130)
    for i, row in enumerate(plan, start=1):
        print(fmt.format(i, str(row.local_exists), row.local_npz_count, row.local_dir))

    print("\nRemote -> Local mapping:")
    for row in plan:
        print(f"REMOTE: {row.remote_dir}")
        print(f"LOCAL:  {row.local_dir}")
        print()


def confirm_transfer(args: argparse.Namespace, n: int) -> bool:
    if args.dry_run:
        return False
    if args.yes:
        return True

    print(f"About to transfer {n} prediction directories. Type YES to continue: ", end="", flush=True)
    ans = input().strip()
    return ans == "YES"


def rsync_one(
    remote_host: str,
    ssh_options: List[str],
    remote_dir: str,
    local_dir: str,
    include_pattern: str,
    timeout_seconds: int,
) -> bool:
    os.makedirs(local_dir, exist_ok=True)

    ssh_inner = "ssh " + " ".join(shlex.quote(opt) for opt in ssh_options)
    src = f"{remote_host}:{remote_dir.rstrip('/')}/"
    dst = f"{local_dir.rstrip('/')}/"

    cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e",
        ssh_inner,
        "--include",
        include_pattern,
        "--exclude",
        "*",
        src,
        dst,
    ]
    try:
        proc = subprocess.run(cmd, timeout=timeout_seconds)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def remote_npz_count(
    remote_host: str,
    ssh_options: List[str],
    remote_dir: str,
    include_pattern: str,
    timeout_seconds: int,
) -> Optional[int]:
    quoted_dir = shlex.quote(remote_dir)
    cmd = (
        "if [ -d {d} ]; then "
        "find {d} -maxdepth 1 -type f -name {g} | wc -l; "
        "else echo __MISSING__; fi"
    ).format(d=quoted_dir, g=shlex.quote(include_pattern))

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


def compute_conflicts(
    plan: List[PathPlan],
    remote_host: str,
    ssh_options: List[str],
    include_pattern: str,
    timeout_seconds: int,
) -> List[ConflictRow]:
    conflicts: List[ConflictRow] = []
    for row in plan:
        if not row.local_exists:
            continue
        rcount = remote_npz_count(
            remote_host=remote_host,
            ssh_options=ssh_options,
            remote_dir=row.remote_dir,
            include_pattern=include_pattern,
            timeout_seconds=timeout_seconds,
        )
        conflicts.append(
            ConflictRow(
                remote_dir=row.remote_dir,
                local_dir=row.local_dir,
                local_npz_count=row.local_npz_count,
                remote_npz_count=rcount,
            )
        )
    return conflicts


def print_conflicts(conflicts: List[ConflictRow]) -> None:
    if not conflicts:
        return

    fmt = "{:<4} {:<10} {:<10} {:<10} {:<90}"
    print("\nConflicts (existing local directories with empty local_suffix):")
    print(fmt.format("#", "local_npz", "remote_npz", "delta", "local_dir"))
    print("-" * 140)
    for i, row in enumerate(conflicts, start=1):
        if row.remote_npz_count is None:
            remote_txt = "unknown"
            delta_txt = "unknown"
        else:
            remote_txt = str(row.remote_npz_count)
            delta_txt = str(row.remote_npz_count - row.local_npz_count)
        print(fmt.format(i, row.local_npz_count, remote_txt, delta_txt, row.local_dir))

    print("\nConflict mapping details:")
    for row in conflicts:
        print(f"REMOTE: {row.remote_dir}")
        print(f"LOCAL:  {row.local_dir}")
        print()


def main() -> None:
    args = parse_args()

    remote_dirs = read_remote_paths(args.paths_file)
    plan = build_plan(
        remote_dirs,
        args.local_prefix,
        args.anchor,
        args.include_pattern,
        args.local_suffix,
    )
    print_plan(plan)

    ssh_options: List[str] = []
    control_path: Optional[str] = None
    started = False

    ok = 0
    fail = 0
    transfer_plan: List[PathPlan] = list(plan)
    skipped_existing: List[PathPlan] = []
    try:
        if not args.local_suffix:
            ssh_options, control_path, started = setup_control_master(args)
            conflicts = compute_conflicts(
                plan=plan,
                remote_host=args.remote_host,
                ssh_options=ssh_options,
                include_pattern=args.include_pattern,
                timeout_seconds=args.ssh_timeout_seconds,
            )
            print_conflicts(conflicts)
            if conflicts and not args.allow_existing_without_suffix:
                conflict_local_dirs = {c.local_dir for c in conflicts}
                skipped_existing = [row for row in plan if row.local_dir in conflict_local_dirs]
                transfer_plan = [row for row in plan if row.local_dir not in conflict_local_dirs]
                print("Conflict protection is active (local_suffix is empty).")
                print("Existing local directories will be skipped.")
                print(
                    "Use --allow_existing_without_suffix if you intentionally want to transfer into existing local directories."
                )

        if args.dry_run:
            if skipped_existing:
                print(f"\n[DRY RUN] Skipping {len(skipped_existing)} existing local directories:")
                for row in skipped_existing:
                    print(f"  - {row.local_dir}")
            print(f"[DRY RUN] Would transfer {len(transfer_plan)} directories.")
            print("\n[DRY RUN] No files transferred.")
            return

        if skipped_existing:
            print(f"\nSkipping {len(skipped_existing)} existing local directories:")
            for row in skipped_existing:
                print(f"  - {row.local_dir}")

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
            print(f"Transferring: {row.remote_dir}")
            print(f"To:          {row.local_dir}")
            print("=" * 80)
            success = rsync_one(
                remote_host=args.remote_host,
                ssh_options=ssh_options,
                remote_dir=row.remote_dir,
                local_dir=row.local_dir,
                include_pattern=args.include_pattern,
                timeout_seconds=args.ssh_timeout_seconds,
            )
            if success:
                ok += 1
            else:
                fail += 1
                print("FAILED transfer for:", row.remote_dir)
    finally:
        teardown_control_master(args.remote_host, control_path, started)

    print("\nTransfer summary")
    print(f"  Success: {ok}")
    print(f"  Failed:  {fail}")
    if skipped_existing:
        print(f"  Skipped existing: {len(skipped_existing)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
