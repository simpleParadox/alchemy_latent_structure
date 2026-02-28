"""
Plot stage/event accuracies from stagewise-accuracy pickle files produced by analyze_predictions.py
(when --save_stagewise_accuracies_only is used).

Expected pickle structure:
  data = {
    "epochs": List[int] or np.ndarray,  # sorted epochs used as x-axis
    "seed_results": {
        0: {  # data_split_seed (usually 0)
            "<metric_key>": List[float] OR Dict[str, List[float]] (reward-binned),
            ...
        }
    }
  }

This script loads one pickle per init_seed, extracts metrics, and plots them on a single figure.

Default mode:
- No averaging unless --plot_mean is set (mean is overlaid + per-seed lines are lightened).

Reward-binning mode (held_out only, --reward_binning_analysis_only):
- Metric is selected via --reward_binning_metric.
- Colors are per reward bin (fixed scheme from analyze_predictions.py).
- If --plot_mean: plot mean only (no individual seed lines), per reward bin.
- Else: plot individual seeds only (no mean), per reward bin.
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# -------------------------
# Where your pickles live
# -------------------------
BASELINE_STAGEWISE_PICKLE_BASEDIR: Dict[str, str] = {
    "composition": "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/stagewise_accuracies_frozen_layer_composition/",
    "decomposition": "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/stagewise_accuracies_frozen_layer_decomposition/",
    "held_out": "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/stagewise_accuracies_frozen_layer_hop_4_exp_held_out/",
}


# -------------------------
# Metrics to plot (default mode)
# -------------------------
COMPOSITION_METRICS: Tuple[str, ...] = (
    "predicted_in_context_accuracies",
    "predicted_in_context_correct_candidate_accuracies",
    "correct_within_candidates",
)

HELDOUT_METRICS: Tuple[str, ...] = (
    "predicted_in_context_accuracies",
    "predicted_in_context_correct_half_accuracies",
    "predicted_in_context_correct_half_exact_accuracies",
    "predicted_in_context_other_half_accuracies",
)

DECOMPOSITION_METRICS: Tuple[str, ...] = (
    "predicted_in_context_accuracies",
    "predicted_in_context_correct_half_accuracies",
    "predicted_in_context_correct_half_exact_accuracies",
    # "predicted_in_adjacent_and_correct_half_accuracies",
    # "predicted_correct_half_within_adjacent_and_correct_half_accuracies"
)

# Custom colors per metric (used in default mode).
METRIC_COLORS: Dict[str, str] = {
    "predicted_in_context_accuracies": "blue",
    "predicted_in_context_correct_candidate_accuracies": "purple",
    "correct_within_candidates": "red",
    "predicted_in_context_correct_half_accuracies": "orange",
    "predicted_in_context_correct_half_exact_accuracies": "green", # Change this based on the hop length.
    "predicted_in_context_other_half_accuracies": "red",

    "predicted_in_adjacent_and_correct_half_accuracies": "cyan",
    "predicted_correct_half_within_adjacent_and_correct_half_accuracies": "pink",
}

# Display labels
CUSTOM_METRICS: Dict[str, Dict[str, str]] = {
    "decomposition": {
        "predicted_in_context_accuracies": "P(A)",
        "predicted_in_context_correct_half_accuracies": "P(B | A)",
        "predicted_in_context_correct_half_exact_accuracies": "P(C | A ∩ B)",
        "predicted_in_adjacent_and_correct_half_accuracies": "P(EN | A)",
        "predicted_correct_half_within_adjacent_and_correct_half_accuracies": "P(NR |EN)",
    },
    "composition": {
        "predicted_in_context_accuracies": "P(A)",
        "predicted_in_context_correct_candidate_accuracies": "P(B | A)",
        "correct_within_candidates": "P(C | A ∩ B)",
    },
    "held_out": {
        "predicted_in_context_accuracies": "P(A) (8 out of 108)",
        "predicted_in_context_correct_half_accuracies": "P(B | A) (4 out of 8)",
        "predicted_in_context_other_half_accuracies": "1 - P(B|A) (4 out of 8)",
        "predicted_in_context_correct_half_exact_accuracies": "P(C|A ∩ B) (1 out of 4)",
    },
}


# -------------------------
# Reward-binning mode config (held_out only)
# -------------------------
REWARD_BIN_COLORS: Dict[str, str] = {
    "-3": "tab:olive",
    "-1": "tab:cyan",
    "1": "tab:pink",
    "3": "tab:brown",
}
REWARD_BIN_LABELS: Dict[str, str] = {
    "-3": "-3",
    "-1": "-1",
    "1": "1",
    "3": "+15",
}

REWARD_BINNING_METRIC_KEY_BY_ARG: Dict[str, str] = {
    "within_support": "within_support_query_stone_state_per_reward_binned_accuracy",
    "within_support_within_half": "within_support_within_half_query_stone_state_per_reward_binned_accuracy",
}
REWARD_BINNING_LINESTYLE_BY_ARG: Dict[str, str] = {
    # Matching the intent in analyze_predictions.py plotting section:
    # within_support was plotted dashed there; within_support_within_half was solid.
    "within_support": "--",
    "within_support_within_half": "-",
}


def _ensure_dir_configured(exp_typ: str) -> str:
    base = BASELINE_STAGEWISE_PICKLE_BASEDIR.get(exp_typ)
    if not base:
        raise ValueError(
            f"Missing base directory for exp_typ='{exp_typ}'. "
            "Edit BASELINE_STAGEWISE_PICKLE_BASEDIR in this file to add it."
        )
    return base


def build_pickle_path(exp_typ: str, hop: int, data_split_seed: int, init_seed: int) -> str:
    """
    Matches analyze_predictions.py save filename (non-frozen):
      stagewise_accuracies_data_split_seed_{data_split_seed}_init_seed_{init_seed}_hop_{hop}_exp_{exp_typ}.pkl
    """
    base = _ensure_dir_configured(exp_typ)
    fname = f"stagewise_accuracies_data_split_seed_{data_split_seed}_init_seed_{init_seed}_hop_{hop}_exp_{exp_typ}.pkl"
    return os.path.join(base, fname)


def load_stagewise_pickle(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def choose_metrics(exp_typ: str) -> Tuple[str, ...]:
    if exp_typ == "composition":
        return COMPOSITION_METRICS
    if exp_typ == "held_out":
        return HELDOUT_METRICS
    if exp_typ == "decomposition":
        return DECOMPOSITION_METRICS
    raise ValueError(f"Unknown exp_typ: {exp_typ}")


def validate_payload(data: dict, data_split_seed: int) -> Tuple[List[int], Dict[str, object]]:
    if "epochs" not in data:
        raise KeyError("Pickle missing key 'epochs'")
    if "seed_results" not in data:
        raise KeyError("Pickle missing key 'seed_results'")

    epochs_raw = data["epochs"]
    if isinstance(epochs_raw, np.ndarray):
        epochs = epochs_raw.tolist()
    else:
        epochs = list(epochs_raw)

    if not epochs:
        raise ValueError("'epochs' must be a non-empty list/array")

    seed_results = data["seed_results"]
    if not isinstance(seed_results, dict):
        raise ValueError("'seed_results' must be a dict")

    # keys might be int or str
    if data_split_seed in seed_results:
        split_payload = seed_results[data_split_seed]
    elif str(data_split_seed) in seed_results:
        split_payload = seed_results[str(data_split_seed)]
    else:
        raise KeyError(
            f"Pickle seed_results missing data_split_seed={data_split_seed}. "
            f"Available keys: {list(seed_results.keys())}"
        )

    if not isinstance(split_payload, dict):
        raise ValueError("seed_results[data_split_seed] must be a dict of metrics")

    return epochs, split_payload


def _truncate_xy(
    epochs: List[int], values: List[float], max_points: Optional[int]
) -> Tuple[List[int], List[float]]:
    if max_points is None or max_points <= 0:
        return epochs, values
    return epochs[:max_points], values[:max_points]


def _lighten_rgba(
    rgba: Tuple[float, float, float, float], amount: float = 0.6
) -> Tuple[float, float, float, float]:
    """
    Lighten an RGBA color by blending it with white.

    amount in [0, 1]:
      0 -> original color
      1 -> white
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    r, g, b, a = rgba
    return (r + (1.0 - r) * amount, g + (1.0 - g) * amount, b + (1.0 - b) * amount, a)


@dataclass(frozen=True)
class SeriesSpec:
    metric: str
    init_seed: int
    epochs: List[int]
    values: List[float]
    reward_bin: Optional[str] = None  # used only in reward-binning mode


def _metric_mean_series_epoch_aligned(series_for_group: Sequence[SeriesSpec], max_points: Optional[int]) -> Tuple[List[int], List[float]]:
    """
    Align by epoch value (not index), then compute nanmean across init_seeds for each epoch.
    Uses union of epochs across seeds; missing values are NaN.
    """
    all_epochs = sorted({e for s in series_for_group for e in s.epochs})
    if max_points is not None and max_points > 0:
        all_epochs = all_epochs[:max_points]
    if not all_epochs:
        return [], []

    epoch_to_vals: Dict[int, List[float]] = {e: [] for e in all_epochs}
    for s in series_for_group:
        ep_to_v = dict(zip(s.epochs, s.values))
        for e in all_epochs:
            epoch_to_vals[e].append(ep_to_v.get(e, np.nan))

    mean_vals = [float(np.nanmean(np.array(epoch_to_vals[e], dtype=float))) for e in all_epochs]
    return all_epochs, mean_vals


def collect_series_default(
    exp_typ: str,
    hop: int,
    data_split_seed: int,
    init_seeds: Sequence[int],
    metrics: Sequence[str],
    strict: bool,
    max_points: Optional[int],
    custom_pkl_path = None,
) -> List[SeriesSpec]:
    series: List[SeriesSpec] = []
    for init_seed in init_seeds:
        if custom_pkl_path is None:
            pkl_path = build_pickle_path(exp_typ=exp_typ, hop=hop, data_split_seed=data_split_seed, init_seed=init_seed)
        else:
            pkl_path = custom_pkl_path
        data = load_stagewise_pickle(pkl_path)
        epochs, payload = validate_payload(data, data_split_seed=data_split_seed)

        for metric in metrics:
            if metric not in payload:
                if strict:
                    raise KeyError(f"Metric '{metric}' missing in {pkl_path}. Available: {list(payload.keys())}")
                print(f"[WARN] metric '{metric}' missing for init_seed={init_seed}; skipping.")
                continue

            values_obj = payload[metric]
            if not isinstance(values_obj, list):
                if strict:
                    raise TypeError(f"Metric '{metric}' is not a list in {pkl_path}; got {type(values_obj)}")
                print(f"[WARN] metric '{metric}' is not a list for init_seed={init_seed}; skipping.")
                continue

            values = values_obj
            n = min(len(values), len(epochs))
            ep, va = _truncate_xy(epochs[:n], values[:n], max_points)
            series.append(SeriesSpec(metric=metric, init_seed=init_seed, epochs=ep, values=va))

    return series


def collect_series_reward_binned(
    hop: int,
    data_split_seed: int,
    init_seeds: Sequence[int],
    reward_metric_key: str,
    strict: bool,
    max_points: Optional[int],
) -> List[SeriesSpec]:
    """
    Collect per-seed series for reward-binned metric:
      payload[reward_metric_key] is expected to be a dict like {'-3': [..], '-1': [..], '1': [..], '3': [..]}
    """
    series: List[SeriesSpec] = []
    for init_seed in init_seeds:
        pkl_path = build_pickle_path(exp_typ="held_out", hop=hop, data_split_seed=data_split_seed, init_seed=init_seed)
        data = load_stagewise_pickle(pkl_path)
        epochs, payload = validate_payload(data, data_split_seed=data_split_seed)

        if reward_metric_key not in payload:
            raise KeyError(f"Reward-binning metric '{reward_metric_key}' missing in {pkl_path}. Available: {list(payload.keys())}")

        obj = payload[reward_metric_key]
        if not isinstance(obj, dict):
            raise TypeError(
                f"Expected dict for reward-binned metric '{reward_metric_key}' in {pkl_path}, got {type(obj)}"
            )

        for rbin in ["-3", "-1", "1", "3"]:
            if rbin not in obj:
                if strict:
                    raise KeyError(f"Reward bin '{rbin}' missing under '{reward_metric_key}' in {pkl_path}. Keys={list(obj.keys())}")
                print(f"[WARN] reward bin '{rbin}' missing for init_seed={init_seed}; skipping.")
                continue

            values_obj = obj[rbin]
            if not isinstance(values_obj, list):
                if strict:
                    raise TypeError(f"Reward bin '{rbin}' under '{reward_metric_key}' is not a list in {pkl_path}.")
                print(f"[WARN] reward bin '{rbin}' is not a list for init_seed={init_seed}; skipping.")
                continue

            values = values_obj
            n = min(len(values), len(epochs))
            ep, va = _truncate_xy(epochs[:n], values[:n], max_points)
            series.append(SeriesSpec(metric=reward_metric_key, init_seed=init_seed, epochs=ep, values=va, reward_bin=rbin))

    return series


# -------------------------
# Plot styling (shared)
# -------------------------
FIGSIZE: Tuple[int, int] = (12, 8)
LEGEND_FONTSIZE: int = 18
SAVE_DPI: int = 300


def plot_default(
    exp_typ: str,
    series: Sequence[SeriesSpec],
    title: str,
    output: Optional[str],
    y_lim: Optional[Tuple[float, float]],
    plot_mean: bool,
    seed_alpha: float,
    seed_lighten: float,
    mean_linewidth: float,
    legend_mode: str,
):
    if not series:
        raise ValueError("No series to plot (check paths/metrics/pickles).")

    metrics_sorted = sorted({s.metric for s in series})
    seeds_sorted = sorted({s.init_seed for s in series})

    cmap = plt.get_cmap("tab10")
    metric_to_color: Dict[str, Tuple[float, float, float, float]] = {}
    for i, m in enumerate(metrics_sorted):
        if m in METRIC_COLORS:
            metric_to_color[m] = mcolors.to_rgba(METRIC_COLORS[m])
        else:
            metric_to_color[m] = cmap(i % 10)

    linestyles = ["-", "--", ":", "-."]
    seed_to_ls = {seed: linestyles[i % len(linestyles)] for i, seed in enumerate(seeds_sorted)}

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for m in metrics_sorted:
        base_color = metric_to_color[m]
        light_color = _lighten_rgba(base_color, amount=seed_lighten)

        series_m = [s for s in series if s.metric == m]

        # per-seed lines
        for seed in seeds_sorted:
            s_list = [s for s in series_m if s.init_seed == seed]
            if not s_list:
                continue
            s0 = s_list[0]
            ax.plot(
                s0.epochs,
                s0.values,
                color=light_color if plot_mean else base_color,
                alpha=seed_alpha if plot_mean else 1.0,
                linestyle=seed_to_ls[seed],
                linewidth=1.75 if plot_mean else 2.0,
                label=(f"{m} | init_seed={seed}" if legend_mode == "all" else None),
            )

        # mean overlay
        if plot_mean:
            mean_epochs, mean_vals = _metric_mean_series_epoch_aligned(series_m, max_points=None)
            disp = CUSTOM_METRICS.get(exp_typ, {}).get(m, m)
            ax.plot(
                mean_epochs,
                mean_vals,
                color=base_color,
                linestyle="-",
                linewidth=mean_linewidth,
                label=(f"{disp}" if legend_mode in {"all", "mean_only"} else None),
            )

    ax.set_xlabel("Epoch", fontsize=26)
    ax.set_ylabel("Accuracy", fontsize=26)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(True)
    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=24)
    if legend_mode != "none":

        # Reorder legend labels to be in a specific order. Used for composition plots.
        if exp_typ == "composition":
            handles, labels = ax.get_legend_handles_labels()
            order = [1,2,0]
            ordered_handles = [handles[idx] for idx in order if idx < len(handles)]
            ordered_labels = [labels[idx] for idx in order if idx < len(labels)]
            ax.legend(ordered_handles, ordered_labels, loc="center right", fontsize=LEGEND_FONTSIZE)
        else:
            ax.legend(loc="center right", fontsize=LEGEND_FONTSIZE)

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=SAVE_DPI, bbox_inches='tight')
        root, ext = os.path.splitext(output)
        if ext.lower() != ".pdf":
            fig.savefig(root + ".pdf", dpi=SAVE_DPI, bbox_inches='tight')
        print(f"[OK] Saved: {output}")
    else:
        plt.show()


def plot_reward_binned(
    series: Sequence[SeriesSpec],
    title: str,
    output: Optional[str],
    y_lim: Optional[Tuple[float, float]],
    plot_mean: bool,
    selected_metric_arg: str,
    max_points: Optional[int],
    mean_linewidth: float,   # NEW: reuse existing style knob
    legend_mode: str,        # NEW: keep legend behavior consistent
):
    """
    Reward-binning plotting rules:
    - If plot_mean: plot mean only (no individual seeds), one line per reward bin.
    - Else: plot individual seeds only, one line per (reward bin, seed).
    """
    if not series:
        raise ValueError("No series to plot (check paths/metrics/pickles).")

    linestyle = REWARD_BINNING_LINESTYLE_BY_ARG[selected_metric_arg]
    seeds_sorted = sorted({s.init_seed for s in series})

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if plot_mean:
        # mean-only per reward bin
        for rbin in ["-3", "-1", "1", "3"]:
            group = [s for s in series if s.reward_bin == rbin]
            if not group:
                continue
            mean_epochs, mean_vals = _metric_mean_series_epoch_aligned(group, max_points=max_points)
            ax.plot(
                mean_epochs,
                mean_vals,
                color=REWARD_BIN_COLORS[rbin],
                linestyle=linestyle,
                linewidth=mean_linewidth,
                label=(f"Query with reward feature {REWARD_BIN_LABELS[rbin]}" if legend_mode != "none" else None),
            )
    else:
        # individual-only per reward bin and seed (same linewidth style as default)
        linestyles = ["-", "--", ":", "-."]
        seed_to_ls = {seed: linestyles[i % len(linestyles)] for i, seed in enumerate(seeds_sorted)}

        for rbin in ["-3", "-1", "1", "3"]:
            for seed in seeds_sorted:
                s_list = [s for s in series if s.reward_bin == rbin and s.init_seed == seed]
                if not s_list:
                    continue
                s0 = s_list[0]
                ax.plot(
                    s0.epochs,
                    s0.values,
                    color=REWARD_BIN_COLORS[rbin],
                    linestyle=seed_to_ls[seed],
                    linewidth=2.0,
                    label=(f"Query with reward feature {REWARD_BIN_LABELS[rbin]} | init_seed={seed}" if legend_mode != "none" else None),
                )

    # ax.set_title(title)
    ax.set_xlabel("Epoch", fontsize=26)
    ax.set_ylabel("Accuracy", fontsize=26)

    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=24)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(True)

    if legend_mode != "none":
        ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE)


    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=SAVE_DPI, bbox_inches='tight')
        root, ext = os.path.splitext(output)
        if ext.lower() != ".pdf":
            fig.savefig(root + ".pdf", dpi=SAVE_DPI, bbox_inches='tight')
        print(f"[OK] Saved: {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot stage/event accuracies from stagewise pickle files."
    )
    parser.add_argument(
        "--exp_typ",
        type=str,
        required=True,
        choices=["held_out", "composition", "decomposition"],
        help="Experiment type.",
    )
    parser.add_argument("--hop", type=int, default=4, help="Hop value (used in filename).")
    parser.add_argument("--data_split_seed", type=int, default=0, help="Data split seed key inside pickle. Typically 0.")
    parser.add_argument("--init_seeds", type=int, nargs="+", default=[1, 3, 42], help="Init seeds to load.")
    parser.add_argument("--output", type=str, default=None, help="Output path (.png recommended). If omitted, shows interactively.")
    parser.add_argument("--strict", action="store_true", help="If set, missing metrics/keys raise instead of being skipped.")

    parser.add_argument("--plot_mean", action="store_true", help="Plot mean across init_seeds (behavior differs in reward-binning mode).")
    parser.add_argument("--legend_mode", type=str, default="all", choices=["all", "mean_only", "none"], help="Default mode legend density.")
    parser.add_argument("--seed_alpha", type=float, default=0.35, help="Default mode: alpha for per-seed lines when --plot_mean is enabled.")
    parser.add_argument("--seed_lighten", type=float, default=0.65, help="Default mode: lighten per-seed lines when --plot_mean is enabled.")
    parser.add_argument("--mean_linewidth", type=float, default=3.0, help="Default mode: linewidth for mean line.")

    parser.add_argument("--max_points", type=int, default=500, help="Plot only the first N points (epochs). Use <=0 to disable.")
    parser.add_argument("--ymin", type=float, default=0.0, help="Y-axis min (use --ymax -1 to disable y-lim).")
    parser.add_argument("--ymax", type=float, default=1.0, help="Y-axis max (use -1 to disable y-lim).")

    parser.add_argument("--custom_pickle_path", type=str, default=None, help="Custom pickle path to use instead of building from exp_typ, hop, seeds.")

    # NEW: reward-binning feature (held_out only)
    parser.add_argument(
        "--reward_binning_analysis_only",
        action="store_true",
        help="Held_out only: plot reward-binned accuracies stored in the pickle.",
    )
    parser.add_argument(
        "--reward_binning_metric",
        type=str,
        default="within_support",
        choices=["within_support", "within_support_within_half"],
        help="Which reward-binned metric to plot (held_out only, requires --reward_binning_analysis_only).",
    )

    args = parser.parse_args()

    if args.reward_binning_analysis_only and args.exp_typ != "held_out":
        raise ValueError("--reward_binning_analysis_only is only supported for --exp_typ held_out.")

    y_lim: Optional[Tuple[float, float]]
    if args.ymax == -1:
        y_lim = None
    else:
        y_lim = (args.ymin, args.ymax)

    max_points = args.max_points if args.max_points and args.max_points > 0 else None

    if args.reward_binning_analysis_only:
        metric_key = REWARD_BINNING_METRIC_KEY_BY_ARG[args.reward_binning_metric]
        series = collect_series_reward_binned(
            hop=args.hop,
            data_split_seed=args.data_split_seed,
            init_seeds=args.init_seeds,
            reward_metric_key=metric_key,
            strict=args.strict,
            max_points=max_points,
        )

        title = (
            f"Reward-binned stages | held_out | hop={args.hop} | data_split_seed={args.data_split_seed} | "
            f"metric={args.reward_binning_metric}"
        )
        plot_reward_binned(
            series=series,
            title=title,
            output=args.output,
            y_lim=y_lim,
            plot_mean=args.plot_mean,
            selected_metric_arg=args.reward_binning_metric,
            max_points=max_points,
            mean_linewidth=args.mean_linewidth,  # NEW
            legend_mode=args.legend_mode,        # NEW
        )
        return

    # Default (non reward-binned)
    metrics = list(choose_metrics(args.exp_typ))
    series = collect_series_default(
        exp_typ=args.exp_typ,
        hop=args.hop,
        data_split_seed=args.data_split_seed,
        init_seeds=args.init_seeds,
        metrics=metrics,
        strict=args.strict,
        max_points=max_points,
        custom_pkl_path=args.custom_pickle_path,
    )

    title = f"Stages from pickles | exp_typ={args.exp_typ} | hop={args.hop} | data_split_seed={args.data_split_seed}"
    plot_default(
        exp_typ=args.exp_typ,
        series=series,
        title=title,
        output=args.output,
        y_lim=y_lim,
        plot_mean=args.plot_mean,
        seed_alpha=args.seed_alpha,
        seed_lighten=args.seed_lighten,
        mean_linewidth=args.mean_linewidth,
        legend_mode=args.legend_mode,
    )


if __name__ == "__main__":
    main()