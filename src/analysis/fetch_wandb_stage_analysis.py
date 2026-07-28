"""
fetch_wandb_stage_analysis.py

Query the W&B project simpleparadox/alchemy-meta-learning and classify
each run by the furthest factorization stage it reached within its epoch
budget, based on the val_epoch_accuracy history.

In addition to the stage-count table (Table 3 of the paper), this script
extracts the *epoch* at which each stage completed, and reports four
timing analyses:

  (A) Censored convergence fraction: the fraction of ALL runs (converged or
      not) that have completed a stage by epoch t. Non-converged runs are
      censored at the epoch budget rather than dropped. The final checkpoint
      column must reproduce Table 3 exactly -- this is the built-in
      consistency check.

      *** INTERNAL CONSISTENCY CHECK ONLY -- DO NOT QUOTE IN THE REBUTTAL. ***
      The cumulative fraction at any checkpoint is approximately
          P(config is capable) x CDF of the shared speed distribution,
      so it is downstream of the success rate. Its agreement with Table 3 is
      what the pool-size confound predicts, not evidence against it.

  (B) Earliest convergence: for each (hop, seed), the minimum stage-completion
      epoch over all hyperparameter configs. This is the protocol actually
      used for the figures in the paper, made explicit and quotable.

      *** CONFOUNDED ACROSS HOPS. *** The minimum over a larger pool of
      converged configs is systematically lower than the minimum over a
      smaller pool, even when the underlying speed distributions are
      identical. Success rates differ non-monotonically across hops
      (29.6 / 48.1 / 70.4 / 29.6), so this statistic cannot separate
      "learned faster" from "converges more often". Reported here for
      transparency and because it is what the paper did -- not as evidence
      of a speed difference.

  (C) Convergence-epoch distribution among converged runs (NEW).
      Median / quartiles instead of the minimum. The median estimates the
      same quantity regardless of how many runs are in the pool, so it is
      not inflated by a higher success rate. This is the pool-size-
      independent comparison.

      CAVEAT to state in any write-up: converged runs are still a
      non-random subset of configs. The median removes the pool-SIZE
      artifact; it does not make the comparison unconditional, because the
      surviving populations may differ in composition as well as size.

  (D) Matched-pool earliest convergence (NEW).
      Keeps the paper's "how quickly can this be learned" framing but
      removes the pool-size advantage: subsample every (hop, seed) down to
      the same number of converged configs k, take the minimum, and repeat.
      If the ordering in (B) survives matching, it is not a pool-size
      artifact.

Filters are defined in wandb_stage_filters.py.

Usage:
    # Quick run-count check (uses summary value only; NO timing analysis):
    python fetch_wandb_stage_analysis.py --dry-run

    # Full analysis (fetches complete val_epoch_accuracy history):
    python fetch_wandb_stage_analysis.py

    # Full analysis + dump per-run CSV (recommended: do this once):
    python fetch_wandb_stage_analysis.py --csv results.csv

    # Re-print all tables from an existing CSV (no W&B fetch, instant):
    python fetch_wandb_stage_analysis.py --from-csv results.csv
"""

import argparse
import csv
import random
import statistics
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

WANDB_ENTITY = "simpleparadox"
WANDB_PROJECT = "alchemy-meta-learning"
# WANDB_PROJECT = "alchemy-length-matched-decomposition"




# Display order for the output tables.
TASKS = [
    # ("held_out",      1, "Withheld potion pair"),
    # ("composition",   2, "Composition 2-hop"),
    # ("composition",   3, "Composition 3-hop"),
    # ("composition",   4, "Composition 4-hop"),
    # ("composition",   5, "Composition 5-hop"),
    ("decomposition", 2, "Decomposition 2-hop"),
    ("decomposition", 3, "Decomposition 3-hop"),
    ("decomposition", 4, "Decomposition 4-hop"),
    ("decomposition", 5, "Decomposition 5-hop"),
]

# Fractions of the epoch budget at which to report cumulative convergence.
# The last one (1.0) must reproduce Table 3.
CHECKPOINT_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

# Number of resamples for the matched-pool analysis (D).
MATCHED_TRIALS = 20000


# ---------------------------------------------------------------------------
# Stage classification helpers
# ---------------------------------------------------------------------------

def stage2_threshold(exp_type, hop):
    """Single source of truth for the Stage 2 threshold."""
    return STAGE2_THRESHOLD.get((exp_type, hop), 0.22)


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


def first_sustained_epoch(pairs, threshold, window):
    """
    Epoch of the FIRST point of the earliest run of `window` consecutive
    values >= threshold.

    This matches the paper's definition:
        t_v = min{ t : a_v(t') >= tau  for all t' in [t, t+P] }

    Returns None if the threshold is never sustained -> the run is CENSORED.
    Censored runs must be counted in the denominator of any fraction, never
    dropped.
    """
    count, start_epoch = 0, None
    for epoch, v in pairs:
        if v >= threshold:
            if count == 0:
                start_epoch = epoch
            count += 1
            if count >= window:
                return start_epoch
        else:
            count, start_epoch = 0, None
    return None


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

    s2_thresh = stage2_threshold(exp_type, hop)

    if exceeds_threshold(acc_values, STAGE3_THRESHOLD, window=SUSTAINED_WINDOW):
        return 3
    if exceeds_threshold(acc_values, s2_thresh, window=SUSTAINED_WINDOW):
        return 2
    if exceeds_threshold(acc_values, STAGE1_LO, window=SUSTAINED_WINDOW):
        return 1
    return 0


def fetch_history(run, metric="val_epoch_accuracy", max_epoch=None):
    """
    Fetch the logged (epoch, val_epoch_accuracy) history up to max_epoch.

    Returns a list of (epoch, value) pairs sorted by epoch.

    IMPORTANT: we keep the epoch numbers rather than relying on list position.
    scan_history does not guarantee one row per epoch with no gaps, so list
    index != epoch in general. Any timing analysis must use the logged epoch.

    train.py logs epoch+1, so epochs are 1-indexed.
    """
    try:
        # history_iter = run.scan_history(keys=["baseline/epoch", "baseline/" + metric])
        history_iter = run.scan_history(keys=["epoch", metric])
    except Exception as e:
        print(f"  [WARN] run {run.id}: could not fetch history - {e}")
        return []

    pairs = []
    for row in history_iter:
        val = row.get(metric)
        epoch = row.get("epoch")

        if val is None or epoch is None:
        # val = row.get('baseline/val_epoch_accuracy')
        # epoch = row.get('baseline/epoch')
            continue
        if max_epoch is not None and epoch > max_epoch:
            break
        pairs.append((int(epoch), float(val)))

    pairs.sort()
    return pairs


def check_contiguity(pairs, label, run_id, budget):
    """
    Warn if the epoch series has gaps or duplicates.

    This matters because SUSTAINED_WINDOW counts *consecutive logged points*.
    If epochs are missing, "3 consecutive points" spans more than 3 epochs and
    the completion epoch means something different from the paper's definition.
    """
    if not pairs:
        return
    epochs = [e for e, _ in pairs]
    if len(set(epochs)) != len(epochs):
        print(f"  [WARN] {label} run {run_id}: duplicate epochs in history.")
    gaps = [b - a for a, b in zip(epochs, epochs[1:]) if b - a != 1]
    if gaps:
        print(f"  [WARN] {label} run {run_id}: non-contiguous epochs "
              f"({len(gaps)} gaps, max gap {max(gaps)}). "
              f"SUSTAINED_WINDOW counts logged points, not epochs.")


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _print_stage_tables(results, tasks):
    """Stage-count table (reproduces Table 3) and stall breakdown."""
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
    for exp_type, hop, label in tasks:
        r = results.get((exp_type, hop))
        if not r or r["total"] == 0:
            continue
        tot = r["total"]
        print(
            f"{label:<28} | "
            f"{tot:>10} | "
            f"{r['s1']:>6} ({100*r['s1']/tot:5.1f}%) | "
            f"{r['s2']:>6} ({100*r['s2']/tot:5.1f}%) | "
            f"{r['s3']:>6} ({100*r['s3']/tot:5.1f}%)"
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
    for exp_type, hop, label in tasks:
        r = results.get((exp_type, hop))
        if not r or r["total"] == 0:
            continue
        print(
            f"{label:<28} | "
            f"{r['total'] - r['s1']:>8} | "
            f"{r['s1'] - r['s2']:>8} | "
            f"{r['s2'] - r['s3']:>8} | "
            f"{r['s3']:>9}"
        )
    print(sep2)


def _print_censored_fraction_table(rows_by_task, tasks, stage_key="t_stage3"):
    """
    ANALYSIS (A): fraction of ALL runs that have completed `stage_key` by
    epoch t. Non-converged runs are censored (counted in the denominator,
    never reaching the threshold), so there is NO selection on convergence.

    The final column is at 100% of the epoch budget and MUST match the
    corresponding column of the stage-count table above. If it does not,
    something is wrong with the extraction -- stop and debug before quoting
    any of these numbers.

    *** DO NOT QUOTE IN THE REBUTTAL -- see module docstring. ***
    """
    print(f"\n\nANALYSIS (A): Cumulative convergence fraction for {stage_key} "
          f"(all runs, censored - no selection on convergence)")
    print("  [INTERNAL CHECK ONLY -- downstream of success rate; do not quote.]\n")

    for exp_type, hop, label in tasks:
        rows = rows_by_task.get((exp_type, hop), [])
        if not rows:
            continue
        budget = EPOCH_BUDGET[exp_type]
        checkpoints = [int(round(f * budget)) for f in CHECKPOINT_FRACTIONS]

        n = len(rows)
        cells = []
        for t in checkpoints:
            k = sum(1 for r in rows
                    if r[stage_key] is not None and r[stage_key] <= t)
            cells.append(f"{100.0 * k / n:5.1f}%")

        cp_header = "  ".join(f"e{t:<5}" for t in checkpoints)
        print(f"{label:<28} (n={n:>3}, budget={budget})")
        print(f"{'':<28}  {cp_header}")
        print(f"{'':<28}  " + "  ".join(f"{c:<6}" for c in cells))
        print()

    print("  NOTE: the last column must equal the Stage-3 percentage in the "
          "stage-count table.\n  If it does not, the extraction is wrong.")


def _print_earliest_convergence_table(rows_by_task, tasks, stage_key="t_stage3"):
    """
    ANALYSIS (B): the protocol actually used for the paper's figures.

    For each (task, seed), take the MINIMUM stage-completion epoch over all
    hyperparameter configs for that seed. Every seed contributes as long as
    at least one of its configs converged, so no seed is dropped.

    Also reports how many configs converged per seed, and flags any seed with
    ZERO converged configs -- that is the only case in which a seed is
    genuinely excluded, and it must be reported explicitly if it occurs.

    *** The n_conv column is the pool size. Compare it across hops before
    reading anything into the earliest-epoch column. ***
    """
    print(f"\n\nANALYSIS (B): Earliest convergence per seed for {stage_key} "
          f"(min over hyperparameter configs)")
    print("  [CONFOUNDED ACROSS HOPS -- the minimum falls as n_conv rises.\n"
          "   Read alongside ANALYSIS (C) and (D).]\n")

    header = (
        f"{'Task / hop length':<28} | {'seed':>5} | {'n_cfg':>5} | "
        f"{'n_conv':>6} | {'earliest epoch':>14}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    summary = {}
    for exp_type, hop, label in tasks:
        rows = rows_by_task.get((exp_type, hop), [])
        if not rows:
            continue
        by_seed = defaultdict(list)
        for r in rows:
            by_seed[r["seed"]].append(r[stage_key])

        per_seed_min = []
        for seed in sorted(by_seed, key=lambda s: (s is None, str(s))):
            ts = by_seed[seed]
            conv = [t for t in ts if t is not None]
            if conv:
                m = min(conv)
                per_seed_min.append(m)
                shown = str(m)
            else:
                shown = "CENSORED"
            print(f"{label:<28} | {str(seed):>5} | {len(ts):>5} | "
                  f"{len(conv):>6} | {shown:>14}")

        if per_seed_min:
            mean = statistics.mean(per_seed_min)
            sd = statistics.stdev(per_seed_min) if len(per_seed_min) > 1 else 0.0
            summary[label] = (mean, sd, len(per_seed_min), len(by_seed))
        print(sep)

    print("\nSummary (mean +/- std of earliest-convergence epoch across seeds):\n")
    for label, (mean, sd, n_used, n_total) in summary.items():
        flag = "" if n_used == n_total else \
               f"   <-- WARNING: {n_total - n_used} seed(s) fully censored and EXCLUDED"
        print(f"  {label:<28} {mean:7.1f} +/- {sd:5.1f}   "
              f"(seeds used: {n_used}/{n_total}){flag}")


def _quartiles(sorted_vals):
    """Q1, Q3 with a graceful fallback for very small samples."""
    n = len(sorted_vals)
    if n >= 4:
        q = statistics.quantiles(sorted_vals, n=4)
        return q[0], q[2]
    return float(sorted_vals[0]), float(sorted_vals[-1])


def _print_converged_distribution_table(rows_by_task, tasks,
                                        stage_key="t_stage3"):
    """
    ANALYSIS (C): distribution of stage-completion epoch among CONVERGED runs.

    Unlike ANALYSIS (B), this takes no minimum, so it is not inflated by
    having more surviving configs. The median estimates the same quantity
    regardless of pool size; a larger pool makes the estimate less noisy but
    does not push it up or down.

    Reported both pooled across seeds and per seed, because the pooled
    median can be dominated by whichever seed happened to converge most
    often.

    CAVEAT for the write-up: converged runs remain a non-random subset of
    configs. This removes the pool-SIZE artifact, not the conditioning.
    """
    print(f"\n\nANALYSIS (C): Convergence-epoch distribution among CONVERGED "
          f"runs for {stage_key}")
    print("  [Pool-size independent. This is the cross-hop comparison to "
          "quote.]\n")

    header = (
        f"{'Task / hop length':<28} | {'n_conv':>6} | {'median':>7} | "
        f"{'Q1':>7} | {'Q3':>7} | {'min':>5} | {'max':>5}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for exp_type, hop, label in tasks:
        rows = rows_by_task.get((exp_type, hop), [])
        if not rows:
            continue
        conv = sorted(r[stage_key] for r in rows if r[stage_key] is not None)
        if not conv:
            print(f"{label:<28} | {0:>6} |  (no converged runs)")
            continue
        med = statistics.median(conv)
        q1, q3 = _quartiles(conv)
        print(f"{label:<28} | {len(conv):>6} | {med:>7.1f} | {q1:>7.1f} | "
              f"{q3:>7.1f} | {conv[0]:>5} | {conv[-1]:>5}")
    print(sep)

    # Per-seed medians, so a single lucky seed cannot drive the pooled value.
    print("\nPer-seed medians among converged runs:\n")
    header2 = (
        f"{'Task / hop length':<28} | {'seed':>5} | {'n_conv':>6} | "
        f"{'median':>7}"
    )
    sep2 = "-" * len(header2)
    print(sep2)
    print(header2)
    print(sep2)
    for exp_type, hop, label in tasks:
        rows = rows_by_task.get((exp_type, hop), [])
        if not rows:
            continue
        by_seed = defaultdict(list)
        for r in rows:
            if r[stage_key] is not None:
                by_seed[r["seed"]].append(r[stage_key])
        for seed in sorted(by_seed, key=lambda s: (s is None, str(s))):
            vals = sorted(by_seed[seed])
            print(f"{label:<28} | {str(seed):>5} | {len(vals):>6} | "
                  f"{statistics.median(vals):>7.1f}")
        print(sep2)


def _print_matched_pool_table(rows_by_task, tasks, stage_key="t_stage3",
                              trials=MATCHED_TRIALS, rng_seed=0):
    """
    ANALYSIS (D): earliest-convergence with the pool size held constant.

    Within each comparison group (exp_type), let k be the smallest number of
    converged configs observed for any (task, seed) in that group. For each
    (task, seed), draw k converged configs at random WITHOUT replacement,
    take the minimum, average across seeds, and repeat `trials` times.

    This keeps the paper's "how quickly can this be learned" framing while
    removing the advantage of having more chances at an early draw. If the
    ordering seen in ANALYSIS (B) survives matching, it is not a pool-size
    artifact; if it collapses, it was.

    Seeds with zero converged configs are excluded and reported.
    """
    print(f"\n\nANALYSIS (D): Matched-pool earliest convergence for {stage_key} "
          f"({trials} resamples)")
    print("  [Same statistic as (B), but every hop/seed subsampled to the "
          "same pool size k.]\n")

    rng = random.Random(rng_seed)

    # Group tasks by exp_type so hops are compared within a task family.
    groups = defaultdict(list)
    for exp_type, hop, label in tasks:
        groups[exp_type].append((exp_type, hop, label))

    for exp_type, group_tasks in groups.items():
        # Collect converged epochs per (task, seed).
        per_task_seed = {}
        for et, hop, label in group_tasks:
            rows = rows_by_task.get((et, hop), [])
            if not rows:
                continue
            by_seed = defaultdict(list)
            for r in rows:
                if r[stage_key] is not None:
                    by_seed[r["seed"]].append(r[stage_key])
            if by_seed:
                per_task_seed[(et, hop, label)] = dict(by_seed)

        if len(per_task_seed) < 2:
            continue

        k = min(len(v) for seeds in per_task_seed.values()
                for v in seeds.values())
        if k < 1:
            continue

        print(f"  [{exp_type}]  matched pool size k = {k} converged configs "
              f"per seed")
        header = (f"{'Task / hop length':<28} | {'seeds':>5} | "
                  f"{'matched min (mean)':>19} | {'unmatched min':>13}")
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)

        for (et, hop, label), by_seed in per_task_seed.items():
            seeds = sorted(by_seed, key=lambda s: (s is None, str(s)))
            unmatched = statistics.mean(min(by_seed[s]) for s in seeds)

            acc = []
            for _ in range(trials):
                mins = []
                for s in seeds:
                    vals = by_seed[s]
                    mins.append(min(rng.sample(vals, k)))
                acc.append(statistics.mean(mins))
            print(f"{label:<28} | {len(seeds):>5} | "
                  f"{statistics.mean(acc):>19.1f} | {unmatched:>13.1f}")
        print(sep)
        print()

    print("  Interpretation: compare the 'matched' column across hops.\n"
          "  If the differences seen in the 'unmatched' column shrink or "
          "vanish\n  once k is held constant, those differences reflected "
          "pool size,\n  not learning speed.")


# ---------------------------------------------------------------------------
# CSV round-trip helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "task", "exp_type", "hop", "run_id", "run_name", "state",
    "seed", "wd", "eta_min", "sched", "dseed", "lr",
    "peak_acc", "stage", "n_points", "t_stage1", "t_stage2", "t_stage3",
]


def _parse_optional_int(s):
    if s is None or s == "" or s == "None":
        return None
    return int(s)


def _coerce_seed(s):
    """
    Keep seeds comparable between the live path (ints from W&B config) and
    the --from-csv path (strings). Without this, grouping and sort order
    differ between the two routes.
    """
    if s is None or s == "" or s == "None":
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        return s


def _load_rows_from_csv(path):
    """Rebuild results + per-task rows from a previously written CSV."""
    results = defaultdict(lambda: {"total": 0, "s1": 0, "s2": 0, "s3": 0})
    rows_by_task = defaultdict(list)
    tasks_seen = []

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            exp_type = row["exp_type"]
            hop = int(row["hop"])
            key = (exp_type, hop)
            stage = int(row["stage"])

            results[key]["total"] += 1
            if stage >= 1:
                results[key]["s1"] += 1
            if stage >= 2:
                results[key]["s2"] += 1
            if stage >= 3:
                results[key]["s3"] += 1

            rows_by_task[key].append({
                "seed": _coerce_seed(row["seed"]),
                "wd": row["wd"],
                "eta_min": row["eta_min"],
                "stage": stage,
                "t_stage1": _parse_optional_int(row.get("t_stage1")),
                "t_stage2": _parse_optional_int(row.get("t_stage2")),
                "t_stage3": _parse_optional_int(row.get("t_stage3")),
            })

            if (exp_type, hop, row["task"]) not in tasks_seen:
                tasks_seen.append((exp_type, hop, row["task"]))

    ordered = [t for t in TASKS if (t[0], t[1]) in rows_by_task]
    return results, rows_by_task, (ordered or tasks_seen)


def _run_all_timing_analyses(rows_by_task, tasks, stage_key, trials):
    _print_censored_fraction_table(rows_by_task, tasks, stage_key)
    _print_earliest_convergence_table(rows_by_task, tasks, stage_key)
    _print_converged_distribution_table(rows_by_task, tasks, stage_key)
    _print_matched_pool_table(rows_by_task, tasks, stage_key, trials=trials)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="Path to write per-run CSV")
    parser.add_argument("--from-csv", default=None, metavar="PATH",
                        help="Read an existing per-run CSV and reprint all "
                             "tables (no W&B fetch)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use summary peak value only (no history fetch). "
                             "Timing analyses are DISABLED in this mode.")
    parser.add_argument("--missing", action="store_true",
                        help="Print which hyperparameter combos are absent")
    parser.add_argument("--stage-key", default="t_stage3",
                        choices=["t_stage1", "t_stage2", "t_stage3"],
                        help="Which stage to use for the timing analyses")
    parser.add_argument("--matched-trials", type=int, default=MATCHED_TRIALS,
                        help="Resamples for the matched-pool analysis (D)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Fast path: re-print from an existing CSV without re-fetching
    # ------------------------------------------------------------------
    if args.from_csv:
        results, rows_by_task, tasks = _load_rows_from_csv(args.from_csv)
        _print_stage_tables(results, tasks)
        _run_all_timing_analyses(rows_by_task, tasks, args.stage_key,
                                 args.matched_trials)
        return

    api = wandb.Api(timeout=120)

    results = defaultdict(lambda: {"total": 0, "s1": 0, "s2": 0, "s3": 0})
    rows_by_task = defaultdict(list)
    csv_rows = []
    tasks_present = []
    dup_report = []

    for exp_type, hop, label in TASKS:
        key = (exp_type, hop)
        if key not in WANDB_FILTERS:
            print(f"[{label}] no filter defined - skipping.")
            continue

        filters = WANDB_FILTERS[key]
        budget = EPOCH_BUDGET[exp_type]
        s2_thresh = stage2_threshold(exp_type, hop)

        print(f"\n[{label}]  fetching runs ...")

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
        if not runs:
            continue

        # --------------------------------------------------------------
        # Deduplicate by full hyperparameter key.
        #
        # NOTE ON DEDUP POLICY: the original script kept the highest-peak
        # duplicate. For a TIMING analysis the consistent choice is the
        # EARLIEST-CONVERGING duplicate, since the estimand is "how quickly
        # can this be learned". We therefore rank by (converged?, t_stage3,
        # -peak) and keep the best. Peak is only a tiebreak.
        # --------------------------------------------------------------
        groups: dict[tuple, dict] = {}
        n_dup = 0

        for i, run in enumerate(runs):
            cfg = run.config
            seed = _coerce_seed(cfg.get("seed"))
            wd = cfg.get("weight_decay")
            eta = cfg.get("eta_min")
            sched = cfg.get("scheduler_type")
            dseed = cfg.get("data_split_seed", 0)
            lr = cfg.get("learning_rate")

            group_key = (seed, wd, eta, sched, dseed, lr)

            if args.dry_run:
                peak = run.summary.get("val_epoch_accuracy", 0.0) or 0.0
                pairs = []
                acc_values = [peak]
                t1 = t2 = t3 = None
            else:
                print(f"  [{i+1}/{len(runs)}] history for {run.id} "
                      f"(seed={seed} wd={wd} eta={eta} sched={sched}) ...",
                      end=" ", flush=True)
                pairs = fetch_history(run, max_epoch=budget)
                check_contiguity(pairs, label, run.id, budget)
                acc_values = [v for _, v in pairs]
                peak = max(acc_values) if acc_values else 0.0
                t1 = first_sustained_epoch(pairs, STAGE1_LO, SUSTAINED_WINDOW)
                t2 = first_sustained_epoch(pairs, s2_thresh, SUSTAINED_WINDOW)
                t3 = first_sustained_epoch(pairs, STAGE3_THRESHOLD,
                                           SUSTAINED_WINDOW)
                print(f"peak={peak:.4f}  n={len(acc_values)}  t3={t3}")

            entry = {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "acc_values": acc_values,
                "n_points": len(acc_values),
                "peak_acc": peak,
                "seed": seed,
                "wd": wd,
                "eta_min": eta,
                "sched": sched,
                "dseed": dseed,
                "lr": lr,
                "t_stage1": t1,
                "t_stage2": t2,
                "t_stage3": t3,
            }

            def rank(e):
                # Lower is better: converged first, then earliest, then
                # highest peak as a tiebreak.
                t = e["t_stage3"]
                return (t is None, t if t is not None else 0, -e["peak_acc"])

            if group_key in groups:
                n_dup += 1
                if rank(entry) < rank(groups[group_key]):
                    groups[group_key] = entry
            else:
                groups[group_key] = entry

            if args.verbose:
                print(f"  {run.id:12s}  seed={seed} wd={wd} eta={eta} "
                      f"sched={str(sched):<16}  peak={peak:.4f}  "
                      f"state={run.state}")

        print(f"  After dedup:  {len(groups)} unique runs "
              f"({n_dup} duplicate(s) discarded)")
        dup_report.append((label, n_dup, len(groups)))

        # --------------------------------------------------------------
        # Missing combo analysis
        # --------------------------------------------------------------
        if args.missing:
            seeds = EXPECTED_SEEDS.get(key, [42, 1, 3])
            expected = set(product(seeds,
                                   EXPECTED_WEIGHT_DECAYS,
                                   EXPECTED_ETA_MINS,
                                   EXPECTED_SCHEDULERS,
                                   EXPECTED_DATA_SEEDS,
                                   EXPECTED_LEARNING_RATES))
            missing = expected - set(groups.keys())
            if missing:
                print(f"\n  *** Missing {len(missing)}/{len(expected)} combos "
                      f"for [{label}]: ***")
                for m in sorted(missing, key=lambda t: (str(t[0]), str(t[1]),
                                                        str(t[2]), str(t[3]))):
                    s_m, wd_m, eta_m, sc_m, ds_m, lr_m = m
                    print(f"    seed={str(s_m):<4} wd={str(wd_m):<7} "
                          f"eta_min={str(eta_m):<10} sched={str(sc_m):<18} "
                          f"data_seed={ds_m} lr={lr_m}")
            else:
                print(f"  OK: all {len(expected)} expected combos present.")

        # --------------------------------------------------------------
        # Classify and aggregate
        # --------------------------------------------------------------
        for entry in groups.values():
            stage = classify_stage(entry["acc_values"], exp_type, hop)

            results[key]["total"] += 1
            if stage >= 1:
                results[key]["s1"] += 1
            if stage >= 2:
                results[key]["s2"] += 1
            if stage >= 3:
                results[key]["s3"] += 1

            rows_by_task[key].append({
                "seed": entry["seed"],
                "wd": entry["wd"],
                "eta_min": entry["eta_min"],
                "stage": stage,
                "t_stage1": entry["t_stage1"],
                "t_stage2": entry["t_stage2"],
                "t_stage3": entry["t_stage3"],
            })

            csv_rows.append({
                "task": label,
                "exp_type": exp_type,
                "hop": hop,
                "run_id": entry["run_id"],
                "run_name": entry["run_name"],
                "state": entry["state"],
                "seed": entry["seed"],
                "wd": entry["wd"],
                "eta_min": entry["eta_min"],
                "sched": entry["sched"],
                "dseed": entry["dseed"],
                "lr": entry["lr"],
                "peak_acc": f"{entry['peak_acc']:.4f}",
                "stage": stage,
                "n_points": entry["n_points"],
                "t_stage1": entry["t_stage1"],
                "t_stage2": entry["t_stage2"],
                "t_stage3": entry["t_stage3"],
            })

        tasks_present.append((exp_type, hop, label))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    _print_stage_tables(results, tasks_present)

    total_dups = sum(d for _, d, _ in dup_report)
    if total_dups:
        print(f"\n[dedup] {total_dups} duplicate run(s) discarded overall "
              f"(earliest-converging kept).")
        print("        Check the stage-count table still matches published "
              "Table 3.")
    else:
        print("\n[dedup] No duplicate hyperparameter keys found - the dedup "
              "policy is moot.")

    if args.dry_run:
        print("\n[--dry-run] Timing analyses skipped: they require the full "
              "history fetch. Re-run without --dry-run.")
    else:
        _run_all_timing_analyses(rows_by_task, tasks_present, args.stage_key,
                                 args.matched_trials)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nPer-run details -> {args.csv}")
        print("Re-print tables instantly with:")
        print(f"  python {__file__.split('/')[-1]} --from-csv {args.csv}")


if __name__ == "__main__":
    main()