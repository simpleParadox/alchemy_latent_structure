"""
fetch_wandb_stage_analysis.py

Query the W&B project simpleparadox/alchemy-meta-learning and classify
each run by the furthest factorization stage it reached within its epoch
budget, based on the val_epoch_accuracy history.

Filters are defined in wandb_stage_filters.py.

Usage:
    # Quick run-count check (uses summary value only, no history fetch):
    python fetch_wandb_stage_analysis.py --dry-run

    # Full analysis (fetches complete val_epoch_accuracy history):
    python fetch_wandb_stage_analysis.py

    # Full analysis + dump per-run CSV:
    python fetch_wandb_stage_analysis.py --csv results.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from itertools import product

import wandb

from wandb_stage_filters import (
    WANDB_FILTERS,
    STAGE1_LO,
    STAGE1_HI,
    STAGE3_THRESHOLD,
    STAGE2_THRESHOLD,
    EPOCH_BUDGET,
    SUSTAINED_WINDOW,
    EXPECTED_SEEDS,
    EXPECTED_WEIGHT_DECAYS,
    EXPECTED_ETA_MINS,
    EXPECTED_SCHEDULERS,
    EXPECTED_LEARNING_RATES,
    EXPECTED_DATA_SEEDS,
)

WANDB_ENTITY  = "simpleparadox"
WANDB_PROJECT = "alchemy-meta-learning"

# Display order for the output table (matches the reviewer's table)
TASKS = [
    # ("held_out",      1, "Withheld potion pair"),
    # ("composition",   2, "Composition 2-hop"),
    # ("composition",   3, "Composition 3-hop"),
    # ("composition",   4, "Composition 4-hop"),
    # ("composition",   5, "Composition 5-hop"),
    ("decomposition", 2, "Decomposition 2-hop"),
    # ("decomposition", 3, "Decomposition 3-hop"),
    # ("decomposition", 4, "Decomposition 4-hop"),
    # ("decomposition", 5, "Decomposition 5-hop"),
]


# ---------------------------------------------------------------------------
# Stage classification helpers
# ---------------------------------------------------------------------------

def exceeds_threshold(values, threshold, window=1):
    """True if `window` consecutive values all >= threshold."""
    count = 0
    for v in values:
        if v >= threshold:
            count += 1
            if count >= window:
                return True
        else:
            count = 0
    return False


def classify_stage(acc_values, exp_type, hop):
    """
    Return the highest stage reached (0-3).
      0 = never left random-level accuracy
      1 = P[A] learned (accuracy crossed into ~12.5 % band)
      2 = P[B|A] (or intermediate for held_out) learned
      3 = P[C|A,B] learned (>95 %)
    """
    if not acc_values:
        return 0

    s2_thresh = STAGE2_THRESHOLD.get((exp_type, hop), 0.22)

    if exceeds_threshold(acc_values, STAGE3_THRESHOLD, window=SUSTAINED_WINDOW):
        return 3
    if exceeds_threshold(acc_values, s2_thresh, window=SUSTAINED_WINDOW):
        return 2
    # For Stage 1: run must cross STAGE1_LO (and also pass STAGE1_LO if it
    # shot straight past to Stage 2+, which is already covered above).
    if exceeds_threshold(acc_values, STAGE1_LO, window=SUSTAINED_WINDOW):
        return 1
    return 0


def fetch_history(run, metric="val_epoch_accuracy", max_epoch=None):
    """
    Fetch the complete logged val_epoch_accuracy history up to max_epoch epochs.

    Uses scan_history with the logged 'epoch' key (train.py logs epoch+1, so
    it is 1-indexed). This matches the pattern from the user's working ipynb:

        history_list = run.scan_history(keys=["epoch", "val_epoch_accuracy"])
        history = list(history_list)[:max_epoch]

    If an explicit max_epoch is given we filter by the logged epoch value so
    we never accidentally include epochs beyond the budget.
    """
    try:
        history_iter = run.scan_history(keys=["epoch", metric])
    except Exception as e:
        print(f"  [WARN] run {run.id}: could not fetch history — {e}")
        return []

    values = []
    for row in history_iter:
        val   = row.get(metric)
        epoch = row.get("epoch")   # logged as epoch+1 (1-indexed) in train.py
        if val is None:
            continue
        # Filter by actual epoch number when a budget is set
        if max_epoch is not None and epoch is not None:
            if epoch > max_epoch:
                break
        values.append(float(val))
    return values


# ---------------------------------------------------------------------------
# Table printing (shared by --from-csv fast path and normal fetch path)
# ---------------------------------------------------------------------------

def _print_tables(results):
    """Print the stage-count table and stall breakdown from a results dict."""
    print("\n")
    header = (
        f"{'Task / hop length':<28} | "
        f"{'Total runs':>10} | "
        f"{'Stage 1 reached':>15} | "
        f"{'Stage 2 reached':>15} | "
        f"{'Stage 3 reached':>15}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for exp_type, hop, label in TASKS:
        r = results[(exp_type, hop)]
        print(
            f"{label:<28} | "
            f"{r['total']:>10} | "
            f"{r['s1']:>15} | "
            f"{r['s2']:>15} | "
            f"{r['s3']:>15}"
        )
    print(sep)

    print("\nStall breakdown (where did runs that failed to converge stop?):\n")
    header2 = (
        f"{'Task / hop length':<28} | "
        f"{'<Stage 1':>8} | "
        f"{'@Stage 1':>8} | "
        f"{'@Stage 2':>8} | "
        f"{'Converged':>9}"
    )
    sep2 = "-" * len(header2)
    print(sep2)
    print(header2)
    print(sep2)
    for exp_type, hop, label in TASKS:
        r = results[(exp_type, hop)]
        n0 = r["total"] - r["s1"]
        n1 = r["s1"]    - r["s2"]
        n2 = r["s2"]    - r["s3"]
        n3 = r["s3"]
        print(
            f"{label:<28} | "
            f"{n0:>8} | "
            f"{n1:>8} | "
            f"{n2:>8} | "
            f"{n3:>9}"
        )
    print(sep2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default=None, help="Path to write per-run CSV")
    parser.add_argument("--from-csv",  default=None, metavar="PATH",
                        help="Read an existing per-run CSV and reprint tables (no W&B fetch)")
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Use summary peak value only (no history fetch)")
    parser.add_argument("--missing",   action="store_true",
                        help="After fetching, print which hyperparameter combos are absent per task")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Fast path: re-print tables from an existing CSV without re-fetching
    # ------------------------------------------------------------------
    if args.from_csv:
        results = defaultdict(lambda: {"total": 0, "s1": 0, "s2": 0, "s3": 0})
        with open(args.from_csv, newline="") as f:
            for row in csv.DictReader(f):
                exp_type = row["exp_type"]
                hop      = int(row["hop"])
                stage    = int(row["stage"])
                key      = (exp_type, hop)
                results[key]["total"] += 1
                if stage >= 1: results[key]["s1"] += 1
                if stage >= 2: results[key]["s2"] += 1
                if stage >= 3: results[key]["s3"] += 1
        _print_tables(results)
        return

    api = wandb.Api(timeout=120)

    # results[(exp_type, hop)] = {total, s1, s2, s3}
    results   = defaultdict(lambda: {"total": 0, "s1": 0, "s2": 0, "s3": 0})
    csv_rows  = []


    for exp_type, hop, label in TASKS:
        key     = (exp_type, hop)
        filters = WANDB_FILTERS[key]
        budget  = EPOCH_BUDGET[exp_type]

        print(f"\n[{label}]  fetching runs …")

        try:
            runs = list(api.runs(
                f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                filters=filters,
                order="+created_at",
            ))
        except Exception as e:
            print(f"  ERROR fetching runs: {e}")
            continue

        print(f"  Raw run count: {len(runs)}")

        # ------------------------------------------------------------------
        # Deduplicate: group by the full hyperparameter key and keep only
        # the run with the highest peak accuracy (handles re-sweeps).
        # Key = (seed, weight_decay, eta_min, scheduler_type, data_split_seed, lr)
        # ------------------------------------------------------------------
        groups: dict[tuple, dict] = {}

        for i, run in enumerate(runs):
            cfg = run.config
            seed  = cfg.get("seed")
            wd    = cfg.get("weight_decay")
            eta   = cfg.get("eta_min")
            sched = cfg.get("scheduler_type")
            dseed = cfg.get("data_split_seed", 0)
            lr    = cfg.get("learning_rate")

            group_key = (seed, wd, eta, sched, dseed, lr)

            # Peak accuracy: from summary (fast) or history (full)
            if args.dry_run:
                peak = run.summary.get("val_epoch_accuracy", 0.0) or 0.0
                acc_values = [peak]
            else:
                print(f"  [{i+1}/{len(runs)}] Fetching history for run {run.id} "
                      f"(seed={seed} wd={wd} eta={eta} sched={sched}) …",
                      end=" ", flush=True)
                acc_values = fetch_history(run, max_epoch=budget)
                peak = max(acc_values) if acc_values else 0.0
                print(f"peak={peak:.4f}  n_points={len(acc_values)}")

            entry = {
                "run_id":     run.id,
                "run_name":   run.name,
                "state":      run.state,
                "acc_values": acc_values,
                "peak_acc":   peak,
                "seed":       seed,
                "wd":         wd,
                "eta_min":    eta,
                "sched":      sched,
                "dseed":      dseed,
            }

            if group_key not in groups or peak > groups[group_key]["peak_acc"]:
                groups[group_key] = entry

            if args.verbose:
                print(f"  {run.id:12s}  seed={seed} wd={wd} eta={eta} "
                      f"sched={sched:<16}  peak={peak:.4f}  state={run.state}")

        print(f"  After dedup:  {len(groups)} unique runs")

        # ------------------------------------------------------------------
        # Missing combo analysis (--missing flag)
        # ------------------------------------------------------------------
        if args.missing:
            seeds    = EXPECTED_SEEDS.get(key, [42, 1, 3])
            expected = set(
                product(seeds,
                        EXPECTED_WEIGHT_DECAYS,
                        EXPECTED_ETA_MINS,
                        EXPECTED_SCHEDULERS,
                        EXPECTED_DATA_SEEDS,
                        EXPECTED_LEARNING_RATES)
            )
            # group_key = (seed, wd, eta, sched, dseed, lr)
            found    = set(groups.keys())
            # Normalise found keys to match expected tuple order
            # (order in group_key: seed, wd, eta, sched, dseed, lr  ← same as product above)
            missing  = expected - found

            if missing:
                print(f"\n  *** Missing {len(missing)}/{len(expected)} combos for [{label}]: ***")
                for m in sorted(missing, key=lambda t: (t[0], t[1], t[2], t[3])):
                    seed_m, wd_m, eta_m, sched_m, dseed_m, lr_m = m
                    print(f"    seed={seed_m:<3}  wd={wd_m:<6}  eta_min={eta_m:<10}  "
                          f"sched={sched_m:<18}  data_seed={dseed_m}  lr={lr_m}")
            else:
                print(f"  ✓ All {len(expected)} expected combos present for [{label}]")

        # ------------------------------------------------------------------
        # Classify and aggregate
        # ------------------------------------------------------------------
        for entry in groups.values():
            stage = classify_stage(entry["acc_values"], exp_type, hop)

            results[key]["total"] += 1
            if stage >= 1: results[key]["s1"] += 1
            if stage >= 2: results[key]["s2"] += 1
            if stage >= 3: results[key]["s3"] += 1

            csv_rows.append({
                "task":     label,
                "exp_type": exp_type,
                "hop":      hop,
                "run_id":   entry["run_id"],
                "run_name": entry["run_name"],
                "state":    entry["state"],
                "seed":     entry["seed"],
                "wd":       entry["wd"],
                "eta_min":  entry["eta_min"],
                "sched":    entry["sched"],
                "dseed":    entry["dseed"],
                "peak_acc": f"{entry['peak_acc']:.4f}",
                "stage":    stage,
            })

    _print_tables(results)

    # -----------------------------------------------------------------------
    # Optional CSV
    # -----------------------------------------------------------------------
    if args.csv:
        fields = ["task", "exp_type", "hop", "run_id", "run_name", "state",
                  "seed", "wd", "eta_min", "sched", "dseed", "peak_acc", "stage"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nPer-run details → {args.csv}")


if __name__ == "__main__":
    main()
