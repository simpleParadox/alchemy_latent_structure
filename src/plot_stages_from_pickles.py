"""
Plot stage/event accuracies from baseline stagewise-accuracy pickle files.

Expected pickle structure:
  data = {
    "epochs": List[int],  # sorted epochs used as x-axis
    "seed_results": {
        0: {  # data_split_seed (usually 0)
            "<metric_key>": List[float],  # same length as epochs
            ...
        }
    }
  }

This script loads one pickle per init_seed, extracts metrics, and plots them on a single figure.
No averaging across seeds; each init_seed is a separate line style, each metric is a separate color.
"""

from __future__ import annotations

import argparse
import os
import re
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


BASELINE_STAGEWISE_PICKLE_BASEDIR: Dict[str, str] = {
    "composition": "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/stagewise_accuracies_frozen_layer_composition/",
    "decomposition": "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/stagewise_accuracies_frozen_layer_decomposition/",
    "held_out": "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/stagewise_accuracies_relative_epoch_frozen_layer_hop_4_exp_held_out/",
}


# -------------------------
# Metrics to plot
# -------------------------
COMPOSITION_METRICS: Tuple[str, ...] = (
    "predicted_in_context_accuracies",
    "predicted_in_context_correct_candidate_accuracies",
    "correct_within_candidates",
)

# One common metric set for held_out + decomposition (edit to match your pickles).
HELDOUT_AND_DECOMPOSITION_METRICS: Tuple[str, ...] = (
    "predicted_in_context_accuracies",
    "predicted_in_context_correct_half_accuracies",
    "predicted_in_context_correct_half_exact_accuracies",
)

# Custom colors per metric (Matplotlib color names or hex).
METRIC_COLORS: Dict[str, str] = {
    "predicted_in_context_accuracies": "orange",  # blue
    "predicted_in_context_correct_candidate_accuracies": "purple",  # orange
    "correct_within_candidates": "blue",  # green

    "predicted_in_context_correct_half_accuracies": "purple",  # red
    "predicted_in_context_correct_half_exact_accuracies": "blue",  # purple
}


def _ensure_dir_configured(exp_typ: str) -> str:
    base = BASELINE_STAGEWISE_PICKLE_BASEDIR.get(exp_typ)
    if not base:
        raise ValueError(
            f"Missing base directory for exp_typ='{exp_typ}'. "
            "Edit BASELINE_STAGEWISE_PICKLE_BASEDIR in this file to add it."
        )
    return base


def build_pickle_path(
    exp_typ: str,
    hop: int,
    data_split_seed: int,
    init_seed: int,
) -> str:

    base = _ensure_dir_configured(exp_typ)
    filename = f'stagewise_accuracies_data_split_seed_{data_split_seed}_init_seed_{init_seed}_hop_{hop}_exp_{exp_typ}.pkl'

    return os.path.join(
        base,
        filename,
    )


def load_stagewise_pickle(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def choose_metrics(exp_typ: str) -> Tuple[str, ...]:
    if exp_typ == "composition":
        return COMPOSITION_METRICS
    if exp_typ in {"held_out", "decomposition"}:
        return HELDOUT_AND_DECOMPOSITION_METRICS
    raise ValueError(f"Unknown exp_typ: {exp_typ}")


def validate_payload(data: dict, data_split_seed: int) -> Tuple[List[int], Dict[str, List[float]]]:
    if "epochs" not in data:
        raise KeyError("Pickle missing key 'epochs'")
    if "seed_results" not in data:
        raise KeyError("Pickle missing key 'seed_results'")

    epochs = data["epochs"]
    # epochs can be either a list or a numpy.ndarray
    if isinstance(epochs, list):
        if not epochs:
            raise ValueError("'epochs' must be a non-empty list")
    else:
        
        if isinstance(epochs, np.ndarray):
            if epochs.size == 0:
                raise ValueError("'epochs' must be a non-empty numpy.ndarray")
            epochs = epochs.tolist()
        else:
            raise ValueError("'epochs' must be a non-empty list or numpy.ndarray")

    seed_results = data["seed_results"]
    if not isinstance(seed_results, dict):
        raise ValueError("'seed_results' must be a dict")

    if data_split_seed not in seed_results:
        # keys might be strings; try that
        if str(data_split_seed) in seed_results:
            split_payload = seed_results[str(data_split_seed)]
        else:
            raise KeyError(
                f"Pickle seed_results missing data_split_seed={data_split_seed}. "
                f"Available keys: {list(seed_results.keys())}"
            )
    else:
        split_payload = seed_results[data_split_seed]

    if not isinstance(split_payload, dict):
        raise ValueError("seed_results[data_split_seed] must be a dict of metrics")

    return epochs, split_payload


@dataclass(frozen=True)
class SeriesSpec:
    metric: str
    init_seed: int
    epochs: List[int]
    values: List[float]


def collect_series(
    exp_typ: str,
    hop: int,
    data_split_seed: int,
    init_seeds: Sequence[int],
    metrics: Sequence[str],
    strict: bool = False,
) -> List[SeriesSpec]:
    """
    Load one pickle per init_seed and collect metric series.

    If strict=False, missing metric keys are skipped with a warning.
    If strict=True, missing metric keys raises.
    """
    series: List[SeriesSpec] = []

    for init_seed in init_seeds:
        pkl_path = build_pickle_path(
            exp_typ=exp_typ,
            hop=hop,
            data_split_seed=data_split_seed,
            init_seed=init_seed,
        )
        data = load_stagewise_pickle(pkl_path)
        epochs, payload = validate_payload(data, data_split_seed=data_split_seed)

        for metric in metrics:
            if metric not in payload:
                if strict:
                    raise KeyError(
                        f"Metric '{metric}' missing in {pkl_path}. Available: {list(payload.keys())}"
                    )
                print(
                    f"[WARN] metric '{metric}' missing in init_seed={init_seed} pickle. "
                    f"Skipping. Path={pkl_path}"
                )
                continue

            values = payload[metric]
            if not isinstance(values, list):
                print(
                    f"[WARN] metric '{metric}' is not a list in init_seed={init_seed}. "
                    f"Skipping. Path={pkl_path}"
                )
                continue
            if len(values) != len(epochs):
                print(
                    f"[WARN] Length mismatch for metric '{metric}' in init_seed={init_seed}: "
                    f"len(values)={len(values)} != len(epochs)={len(epochs)}. "
                    "Will plot min length."
                )
                n = min(len(values), len(epochs))
                series.append(
                    SeriesSpec(metric=metric, init_seed=init_seed, epochs=epochs[:n], values=values[:n])
                )
            else:
                series.append(
                    SeriesSpec(metric=metric, init_seed=init_seed, epochs=epochs, values=values)
                )

    return series


def _lighten_rgba(rgba: Tuple[float, float, float, float], amount: float = 0.6) -> Tuple[float, float, float, float]:
    """
    Lighten a matplotlib RGBA color by blending it with white.

    amount in [0, 1]:
      0 -> original color
      1 -> white
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    r, g, b, a = rgba
    r2 = r + (1.0 - r) * amount
    g2 = g + (1.0 - g) * amount
    b2 = b + (1.0 - b) * amount
    return (r2, g2, b2, a)


def _metric_mean_series(series_for_metric: Sequence[SeriesSpec]) -> Tuple[List[int], List[float]]:
    """
    Align by epoch value (not index), then compute nanmean across seeds for each epoch.
    Uses the union of epochs across seeds; missing values are treated as NaN.
    """
    all_epochs = sorted({e for s in series_for_metric for e in s.epochs})
    if not all_epochs:
        return [], []

    # epoch -> list of values (one per seed, possibly missing)
    epoch_to_vals: Dict[int, List[float]] = {e: [] for e in all_epochs}

    for s in series_for_metric:
        ep_to_v = dict(zip(s.epochs, s.values))
        for e in all_epochs:
            v = ep_to_v.get(e, np.nan)
            epoch_to_vals[e].append(v)

    mean_vals: List[float] = []
    for e in all_epochs:
        arr = np.array(epoch_to_vals[e], dtype=float)
        mean_vals.append(float(np.nanmean(arr)))

    return all_epochs, mean_vals


def plot_series_one_figure(
    series: Sequence[SeriesSpec],
    title: str,
    output: Optional[str] = None,
    y_lim: Optional[Tuple[float, float]] = (0.0, 1.0),
    figsize: Tuple[int, int] = (10, 6),
    plot_mean: bool = False,
    seed_alpha: float = 0.35,
    seed_lighten: float = 0.65,
    mean_linewidth: float = 3.0,
    legend_mode: str = "all",
):
    if not series:
        raise ValueError("No series to plot (check paths/metrics/pickles).")

    if legend_mode not in {"all", "mean_only", "none"}:
        raise ValueError("legend_mode must be one of: all, mean_only, none")

    # Stable ordering: metrics then init_seed
    metrics_sorted = sorted({s.metric for s in series})
    seeds_sorted = sorted({s.init_seed for s in series})

    # Map metric -> color (prefer user-defined; fallback to tab10)
    cmap = plt.get_cmap("tab10")
    metric_to_color: Dict[str, Tuple[float, float, float, float]] = {}
    for i, m in enumerate(metrics_sorted):
        if m in METRIC_COLORS:
            metric_to_color[m] = mcolors.to_rgba(METRIC_COLORS[m])
        else:
            metric_to_color[m] = cmap(i % 10)

    # Map init_seed -> linestyle (consistent across metrics)
    linestyles = ["-", "--", ":", "-."]
    seed_to_ls: Dict[int, str] = {}
    for i, seed in enumerate(seeds_sorted):
        seed_to_ls[seed] = linestyles[i % len(linestyles)]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot per-metric blocks
    for m in metrics_sorted:
        base_color = metric_to_color[m]
        light_color = _lighten_rgba(base_color, amount=seed_lighten)

        series_m = [s for s in series if s.metric == m]

        # 1) per-seed lines (lighter)
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
                linewidth=2.0 if not plot_mean else 1.75,
                label=(f"{m} | init_seed={seed}" if legend_mode == "all" else None),
            )

        # 2) mean line (thicker, base color)
        if plot_mean:
            mean_epochs, mean_vals = _metric_mean_series(series_m)
            ax.plot(
                mean_epochs,
                mean_vals,
                color=base_color,
                alpha=1.0,
                linestyle="-",
                linewidth=mean_linewidth,
                label=(f"{m} | mean" if legend_mode in {"all", "mean_only"} else None),
            )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy / Value")
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(True, linestyle=":", alpha=0.6)

    if legend_mode != "none":
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f" Saved: {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot stage/event accuracies from baseline stagewise pickle files (no averaging)."
    )
    parser.add_argument(
        "--exp_typ",
        type=str,
        required=True,
        choices=["held_out", "composition", "decomposition"],
        help="Experiment type.",
    )
    parser.add_argument(
        "--hop",
        type=int,
        default=2,
        help="Hop value used in path construction (still required for held_out under default layout).",
    )
    parser.add_argument(
        "--data_split_seed",
        type=int,
        default=0,
        help="Data split seed (key inside pickle seed_results). Typically 0.",
    )
    parser.add_argument(
        "--init_seeds",
        type=int,
        nargs="+",
        default=[1, 3, 42],
        help="Init seeds to load (one pickle per init_seed).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path to save the plot (png/pdf). If omitted, shows interactively.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="If set, missing metrics will raise instead of being skipped.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=0.0,
        help="Y-axis min (set to anything; use --ymax -1 to disable y-lim).",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=1.0,
        help="Y-axis max (use -1 to disable y-lim).",
    )
    parser.add_argument(
        "--plot_mean",
        action="store_true",
        help="If set, overlays a per-metric mean line (thick) and draws per-seed lines as lighter shades.",
    )
    parser.add_argument(
        "--legend_mode",
        type=str,
        default="all",
        choices=["all", "mean_only", "none"],
        help="Legend entries to show.",
    )
    parser.add_argument(
        "--seed_alpha",
        type=float,
        default=0.35,
        help="Alpha for per-seed lines when --plot_mean is enabled.",
    )
    parser.add_argument(
        "--seed_lighten",
        type=float,
        default=0.65,
        help="How much to lighten per-seed lines when --plot_mean is enabled (0=none, 1=white).",
    )
    parser.add_argument(
        "--mean_linewidth",
        type=float,
        default=3.0,
        help="Linewidth of mean line when --plot_mean is enabled.",
    )

    args = parser.parse_args()

    metrics = list(choose_metrics(args.exp_typ))

    # Collect series from pickles
    series = collect_series(
        exp_typ=args.exp_typ,
        hop=args.hop,
        data_split_seed=args.data_split_seed,
        init_seeds=args.init_seeds,
        metrics=metrics,
        strict=args.strict,
    )

    y_lim: Optional[Tuple[float, float]]
    if args.ymax == -1:
        y_lim = None
    else:
        y_lim = (args.ymin, args.ymax)

    title = f"Stages from pickles | exp_typ={args.exp_typ} | hop={args.hop} | data_split_seed={args.data_split_seed}"
    plot_series_one_figure(
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