"""
plot_patching_results_multiseed.py

Visualizes activation patching results averaged across multiple seeds.
Reads a JSON config file containing result directories for all hops,
and uses the --hop argument to select which hop to plot.

Produces:
  - Attention head line plots: one subplot per layer, 4 heads per subplot.
    Each head is a colored line (mean across seeds) with SEM shaded band.
  - MLP + embedding line plot: same treatment.

Usage:
  python src/mech_interp/plot_patching_results_multiseed.py \
      --config configs/plot_multiseed_composition.json --hop 2
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for clusters)
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SCALAR_KEYS = {"softmax_mean", "lse_mean", "raw_lse_mean"}


def load_single_seed(results_dir: str, setup: str) -> dict:
    """
    Load and merge all epoch-range pickle files for a single seed directory.

    Returns a dict: epoch (int) -> component_name -> metric_dict.
    """
    pattern = os.path.join(results_dir, f"layer_head_sweep_results_epochs_*_{setup}.pkl")
    pkl_files = sorted(glob.glob(pattern))

    if not pkl_files:
        raise FileNotFoundError(
            f"No pickle files found matching '{pattern}'.\n"
            f"Check results_dir and setup arguments."
        )

    merged: dict = {}
    for path in pkl_files:
        with open(path, "rb") as f:
            data = pickle.load(f)
        for epoch, comp_dict in data.items():
            if epoch not in merged:
                merged[epoch] = {
                    comp: {k: v for k, v in metrics.items() if k in SCALAR_KEYS}
                    for comp, metrics in comp_dict.items()
                }
    return merged


def load_all_seeds(results_dirs: list, setup: str) -> list:
    """
    Load results from multiple seed directories.

    Returns a list of dicts, one per seed.
    """
    all_seeds = []
    for i, rdir in enumerate(results_dirs):
        print(f"  [Seed {i+1}/{len(results_dirs)}] Loading from: {rdir}")
        seed_data = load_single_seed(rdir, setup)
        epochs = sorted(seed_data.keys())
        print(f"    -> Loaded {len(epochs)} epochs: {epochs[0]} ... {epochs[-1]}")
        all_seeds.append(seed_data)
    return all_seeds


# ---------------------------------------------------------------------------
# Alignment and aggregation
# ---------------------------------------------------------------------------

def find_common_epochs(all_seeds: list) -> list:
    """
    Find epochs present in ALL seeds. Warn about missing ones.
    """
    epoch_sets = [set(seed.keys()) for seed in all_seeds]
    common = sorted(set.intersection(*epoch_sets))
    all_epochs = sorted(set.union(*epoch_sets))

    if len(common) < len(all_epochs):
        missing_count = len(all_epochs) - len(common)
        print(f"\n  WARNING: {missing_count} epoch(s) are not present in all seeds "
              f"and will be excluded from plots.")
        for i, es in enumerate(epoch_sets):
            diff = sorted(set(all_epochs) - es)
            if diff:
                print(f"    Seed {i+1}: missing epochs {diff[:10]}{'...' if len(diff) > 10 else ''}")
    else:
        print(f"\n  All {len(common)} epochs are present in every seed.")

    return common


def get_components(all_seeds: list) -> list:
    """
    Get the list of component names from the first epoch of the first seed.
    """
    sample_epoch = next(iter(all_seeds[0]))
    return list(all_seeds[0][sample_epoch].keys())


def compute_mean_sem(all_seeds: list, common_epochs: list, metric: str):
    """
    Compute per-component, per-epoch mean and SEM across seeds.

    Returns:
        means: dict[component] -> np.array of shape (n_epochs,)
        sems:  dict[component] -> np.array of shape (n_epochs,)
    """
    components = get_components(all_seeds)
    n_seeds = len(all_seeds)
    n_epochs = len(common_epochs)

    means = {}
    sems = {}

    for comp in components:
        # [n_seeds, n_epochs] matrix
        scores = np.full((n_seeds, n_epochs), np.nan)
        for si, seed_data in enumerate(all_seeds):
            for ei, ep in enumerate(common_epochs):
                val = seed_data.get(ep, {}).get(comp, {}).get(metric, np.nan)
                scores[si, ei] = val

        means[comp] = np.nanmean(scores, axis=0)
        sems[comp] = np.nanstd(scores, axis=0, ddof=1) / np.sqrt(n_seeds)

    return means, sems


# ---------------------------------------------------------------------------
# Component parsing
# ---------------------------------------------------------------------------

def parse_components(components: list):
    """
    Return grouped component lists.
    """
    attn_heads = sorted(
        [c for c in components if re.match(r"layer_\d+_head_\d+", c)],
        key=lambda c: (int(re.search(r"layer_(\d+)", c).group(1)),
                       int(re.search(r"head_(\d+)", c).group(1)))
    )
    mlps = sorted(
        [c for c in components if c.endswith("_mlp_out")],
        key=lambda c: int(re.search(r"layer_(\d+)", c).group(1))
    )
    embedding = [c for c in components if c == "embedding"]
    return attn_heads, mlps, embedding


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth(vals, window: int):
    """Apply a simple rolling average."""
    if window <= 1:
        return vals
    arr = np.array(vals, dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="same")
    half = window // 2
    smoothed[:half] = np.nan
    smoothed[-half:] = np.nan
    return smoothed


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_attention_heads(common_epochs, means, sems, setup, metric, output_dir,
                         attn_heads, smoothing_window, clip_percentile,
                         experiment, hop, n_seeds, ylim_list=None, sharey=False):
    """
    One subplot per layer, each with all heads as separate colored lines.
    Mean line + SEM shaded band. Raw mean faintly behind smoothed trend.
    """
    layer_ids = sorted(set(
        int(re.search(r"layer_(\d+)", c).group(1)) for c in attn_heads
    ))
    n_layers = len(layer_ids)
    epochs_arr = np.array(common_epochs)

    # Handle ylim_list parsing and expansion
    if ylim_list is not None:
        if len(ylim_list) == 1:
            # Replicate the single ylim for all layers
            ylim_list = ylim_list * n_layers
        elif len(ylim_list) != n_layers:
            print(f"Warning: Length of ylim_list ({len(ylim_list)}) does not match n_layers ({n_layers}). Using default limits.")
            ylim_list = None

    # If different limits are specified, sharey must be False
    if ylim_list is not None:
        if any(ylim_list[i] != ylim_list[0] for i in range(len(ylim_list))):
            sharey = False

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4.5), sharey=sharey)
    if n_layers == 1:
        axes = [axes]
    fig.suptitle(
        f"Attention Head Patching — {experiment} | hop {hop} | {setup} | {metric}\n"
        f"Mean ± SEM across {n_seeds} seeds",
        fontsize=12
    )

    # Compute global default y-limits if sharing y
    if sharey and ylim_list is None:
        all_vals = []
        for comp in attn_heads:
            all_vals.extend(means[comp].tolist())
        if clip_percentile < 100 and all_vals:
            limit = float(np.nanpercentile(np.abs(all_vals), clip_percentile))
        else:
            limit = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals))) if all_vals else 1.0
        default_ylim = (-limit, limit) if limit > 0 else None
    else:
        default_ylim = None

    cmap = plt.get_cmap("tab10")
    for li, (ax, layer) in enumerate(zip(axes, layer_ids)):
        layer_comps = [c for c in attn_heads if f"layer_{layer}_" in c]
        for i, comp in enumerate(layer_comps):
            head_id = re.search(r"head_(\d+)", comp).group(1)
            color = cmap(i)
            mean_vals = means[comp]
            sem_vals = sems[comp]

            # Raw mean: faint
            ax.plot(epochs_arr, mean_vals, color=color, linewidth=0.5, alpha=0.2)
            # Smoothed mean: bold
            smoothed_mean = smooth(mean_vals, smoothing_window)
            ax.plot(epochs_arr, smoothed_mean, color=color, linewidth=1.8,
                    label=f"head {head_id}")
            # SEM band around smoothed line
            smoothed_sem = smooth(sem_vals, smoothing_window)
            ax.fill_between(epochs_arr,
                            smoothed_mean - smoothed_sem,
                            smoothed_mean + smoothed_sem,
                            color=color, alpha=0.15)

        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Epoch")
        if layer == layer_ids[0]:
            ax.set_ylabel(metric)

        # Apply y-limit for this specific subplot
        if ylim_list is not None:
            current_ylim = ylim_list[li]
        elif sharey:
            current_ylim = default_ylim
        else:
            # Compute independent autoscaled y-limit for this specific layer
            all_vals = []
            for comp in layer_comps:
                all_vals.extend(means[comp].tolist())
            if clip_percentile < 100 and all_vals:
                limit = float(np.nanpercentile(np.abs(all_vals), clip_percentile))
            else:
                limit = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals))) if all_vals else 1.0
            current_ylim = (-limit, limit) if limit > 0 else None

        if current_ylim:
            ax.set_ylim(current_ylim)


        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"multiseed_line_attn_{experiment}_hop{hop}_{setup}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_mlps_embedding(common_epochs, means, sems, setup, metric, output_dir,
                        mlps, embedding, smoothing_window, clip_percentile,
                        experiment, hop, n_seeds, ylim=None):
    """
    MLP + embedding line plot with SEM bands.
    """
    mlp_emb = embedding + mlps
    epochs_arr = np.array(common_epochs)

    # Compute default y-limits if not provided
    if ylim is None:
        all_vals = []
        for comp in mlp_emb:
            all_vals.extend(means[comp].tolist())
        if clip_percentile < 100 and all_vals:
            limit = float(np.nanpercentile(np.abs(all_vals), clip_percentile))
        else:
            limit = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals))) if all_vals else 1.0
        ylim = (-limit, limit) if limit > 0 else None

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle(
        f"MLP & Embedding Patching — {experiment} | hop {hop} | {setup} | {metric}\n"
        f"Mean ± SEM across {n_seeds} seeds",
        fontsize=12
    )

    cmap = plt.get_cmap("tab10")
    for i, comp in enumerate(mlp_emb):
        color = cmap(i)
        mean_vals = means[comp]
        sem_vals = sems[comp]

        # Raw mean: faint
        ax.plot(epochs_arr, mean_vals, color=color, linewidth=0.5, alpha=0.2)
        # Smoothed mean: bold
        smoothed_mean = smooth(mean_vals, smoothing_window)
        ax.plot(epochs_arr, smoothed_mean, color=color, linewidth=1.8,
                label=comp)
        # SEM band
        smoothed_sem = smooth(sem_vals, smoothing_window)
        ax.fill_between(epochs_arr,
                        smoothed_mean - smoothed_sem,
                        smoothed_mean + smoothed_sem,
                        color=color, alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"multiseed_line_mlp_{experiment}_hop{hop}_{setup}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_attention_heads_comparison(common_epochs, means_n, means_d, metric, output_dir,
                                    attn_heads, smoothing_window, clip_percentile,
                                    experiment, hop, n_seeds, ylim_list=None, sharey=False):
    """
    Comparison plot for noising vs denoising. 
    Solid line = noising, dashed line = denoising. SEM is omitted.
    """
    layer_ids = sorted(set(
        int(re.search(r"layer_(\d+)", c).group(1)) for c in attn_heads
    ))
    n_layers = len(layer_ids)
    epochs_arr = np.array(common_epochs)

    if ylim_list is not None:
        if len(ylim_list) == 1:
            ylim_list = ylim_list * n_layers
        elif len(ylim_list) != n_layers:
            print(f"Warning: Length of ylim_list ({len(ylim_list)}) does not match n_layers ({n_layers}). Using default limits.")
            ylim_list = None
    if ylim_list is not None and any(ylim_list[i] != ylim_list[0] for i in range(len(ylim_list))):
        sharey = False

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4.5), sharey=sharey)
    if n_layers == 1:
        axes = [axes]
    fig.suptitle(
        f"Attention Head Patching Comparison (Solid=Noising, Dashed=Denoising) — {experiment} | hop {hop} | {metric}\n"
        f"Mean across {n_seeds} seeds",
        fontsize=12
    )

    if sharey and ylim_list is None:
        all_vals = []
        for comp in attn_heads:
            all_vals.extend(means_n[comp].tolist() + means_d[comp].tolist())
        if clip_percentile < 100 and all_vals:
            limit = float(np.nanpercentile(np.abs(all_vals), clip_percentile))
        else:
            limit = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals))) if all_vals else 1.0
        default_ylim = (-limit, limit) if limit > 0 else None
    else:
        default_ylim = None

    cmap = plt.get_cmap("tab10")
    for li, (ax, layer) in enumerate(zip(axes, layer_ids)):
        layer_comps = [c for c in attn_heads if f"layer_{layer}_" in c]
        for i, comp in enumerate(layer_comps):
            head_id = re.search(r"head_(\d+)", comp).group(1)
            color = cmap(i)

            mean_vals_n = means_n[comp]
            mean_vals_d = means_d[comp]

            smoothed_mean_n = smooth(mean_vals_n, smoothing_window)
            smoothed_mean_d = smooth(mean_vals_d, smoothing_window)

            # Plot noising (solid)
            ax.plot(epochs_arr, smoothed_mean_n, color=color, linestyle='-', linewidth=1.5, label=f"head {head_id} (N)")
            # Plot denoising (dashed)
            ax.plot(epochs_arr, smoothed_mean_d, color=color, linestyle='--', linewidth=1.5, label=f"head {head_id} (D)")

        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Epoch")
        if layer == layer_ids[0]:
            ax.set_ylabel(metric)

        if ylim_list is not None:
            current_ylim = ylim_list[li]
        elif sharey:
            current_ylim = default_ylim
        else:
            all_vals = []
            for comp in layer_comps:
                all_vals.extend(means_n[comp].tolist() + means_d[comp].tolist())
            if clip_percentile < 100 and all_vals:
                limit = float(np.nanpercentile(np.abs(all_vals), clip_percentile))
            else:
                limit = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals))) if all_vals else 1.0
            current_ylim = (-limit, limit) if limit > 0 else None

        if current_ylim:
            ax.set_ylim(current_ylim)

        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"multiseed_line_attn_{experiment}_hop{hop}_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_mlps_embedding_comparison(common_epochs, means_n, means_d, metric, output_dir,
                                   mlps, embedding, smoothing_window, clip_percentile,
                                   experiment, hop, n_seeds, ylim=None):
    """
    Comparison plot for MLPs and Embeddings. 
    Solid line = noising, dashed line = denoising. SEM is omitted.
    """
    mlp_emb = embedding + mlps
    epochs_arr = np.array(common_epochs)

    if ylim is None:
        all_vals = []
        for comp in mlp_emb:
            all_vals.extend(means_n[comp].tolist() + means_d[comp].tolist())
        if clip_percentile < 100 and all_vals:
            limit = float(np.nanpercentile(np.abs(all_vals), clip_percentile))
        else:
            limit = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals))) if all_vals else 1.0
        ylim = (-limit, limit) if limit > 0 else None

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle(
        f"MLP & Embedding Patching Comparison (Solid=Noising, Dashed=Denoising) — {experiment} | hop {hop} | {metric}\n"
        f"Mean across {n_seeds} seeds",
        fontsize=12
    )

    cmap = plt.get_cmap("tab10")
    for i, comp in enumerate(mlp_emb):
        color = cmap(i)

        mean_vals_n = means_n[comp]
        mean_vals_d = means_d[comp]

        smoothed_mean_n = smooth(mean_vals_n, smoothing_window)
        smoothed_mean_d = smooth(mean_vals_d, smoothing_window)

        ax.plot(epochs_arr, smoothed_mean_n, color=color, linestyle='-', linewidth=1.5, label=f"{comp} (N)")
        ax.plot(epochs_arr, smoothed_mean_d, color=color, linestyle='--', linewidth=1.5, label=f"{comp} (D)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"multiseed_line_mlp_{experiment}_hop{hop}_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Helper functions for y-limit parsing
# ---------------------------------------------------------------------------

def parse_ylim_single(val_str):
    if not val_str:
        return None
    val_str = val_str.replace("[", "").replace("]", "")
    subparts = [float(x) for x in val_str.split(",") if x.strip()]
    if len(subparts) != 2:
        raise argparse.ArgumentTypeError(f"y-limit must have exactly 2 numbers (ymin, ymax). Got: {val_str}")
    return subparts


def parse_ylim_list(val_str):
    if not val_str:
        return None
    val_str = val_str.replace("[", "").replace("]", "")
    parts = val_str.split(";")
    result = []
    for part in parts:
        subparts = [float(x) for x in part.split(",") if x.strip()]
        if len(subparts) != 2:
            raise argparse.ArgumentTypeError(f"Each y-limit must have exactly 2 numbers (ymin, ymax). Got: {part}")
        result.append(subparts)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed activation patching visualization. "
                    "Reads a JSON config and uses --hop to select which hop to plot."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to JSON config file."
    )
    parser.add_argument(
        "--hop", type=int, required=True,
        help="Hop length to plot (e.g. 2, 3, 4, 5)."
    )
    # Allow CLI overrides for common settings
    parser.add_argument("--setup", type=str, default=None,
                        choices=["noising", "denoising", "comparison"])
    parser.add_argument("--metric", type=str, default=None,
                        choices=["lse_mean", "softmax_mean", "raw_lse_mean"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--smoothing_window", type=int, default=None)
    parser.add_argument("--clip_percentile", type=float, default=None)
    parser.add_argument(
        "--attn_ylim", type=str, default=None,
        help="y-limits for attention head subplots. Can be a single range like '-1,1' "
             "or a semicolon-separated list of ranges like '-0.5,0.5;-1,1;-1.5,1.5;-2,2'."
    )
    parser.add_argument(
        "--sharey", type=str, default="False", choices=["True", "False"],
        help="Whether to share the y-axis limit/scale across all attention head subplots. Default is False."
    )
    parser.add_argument(
        "--mlp_ylim", type=str, default=None,
        help="y-limit range for MLP/embedding plot, e.g. '-1,1'."
    )

    args = parser.parse_args()

    # Load JSON config
    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Look up the requested hop
    hop = args.hop
    hop_key = str(hop)
    if "hops" not in cfg or hop_key not in cfg["hops"]:
        available = list(cfg.get("hops", {}).keys())
        print(f"ERROR: Hop {hop} not found in config. Available hops: {available}")
        sys.exit(1)

    hop_cfg = cfg["hops"][hop_key]
    results_dirs = hop_cfg["results_dirs"]

    # Merge CLI overrides with config (CLI takes precedence)
    experiment = cfg.get("experiment", "unknown")
    setup = args.setup or cfg.get("setup", "noising")
    metric = args.metric or cfg.get("metric", "raw_lse_mean")
    output_base_dir = args.output_dir or cfg.get("output_base_dir", "./plots")
    output_dir = os.path.join(output_base_dir, f"hop{hop}_{experiment}")
    smoothing_window = args.smoothing_window or cfg.get("smoothing_window", 25)
    clip_percentile = args.clip_percentile if args.clip_percentile is not None else cfg.get("clip_percentile", 100.0)

    # Merge y-limits from CLI / config
    attn_ylim = None
    if args.attn_ylim is not None:
        attn_ylim = parse_ylim_list(args.attn_ylim)
    elif "attn_ylim" in cfg:
        raw_attn_ylim = cfg["attn_ylim"]
        if isinstance(raw_attn_ylim, list):
            if all(isinstance(x, list) for x in raw_attn_ylim):
                attn_ylim = raw_attn_ylim
            else:
                attn_ylim = [raw_attn_ylim]

    mlp_ylim = None
    if args.mlp_ylim is not None:
        mlp_ylim = parse_ylim_single(args.mlp_ylim)
    elif "mlp_ylim" in cfg:
        mlp_ylim = cfg["mlp_ylim"]

    os.makedirs(output_dir, exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("Multi-seed Patching Results Plotter")
    print("=" * 70)
    print(f"Experiment : {experiment}")
    print(f"Hop        : {hop}")
    print(f"Setup      : {setup}")
    print(f"Metric     : {metric}")
    print(f"Smoothing  : {smoothing_window}")
    print(f"Clip %ile  : {clip_percentile}")
    print(f"Attn ylim  : {attn_ylim}")
    print(f"MLP ylim   : {mlp_ylim}")
    print(f"Output dir : {output_dir}")
    print(f"Num seeds  : {len(results_dirs)}")
    print("-" * 70)
    print("Loading result directories:")

    if setup == "comparison":
        print("\nLoading result directories for both noising and denoising:")
        all_seeds_n = load_all_seeds(results_dirs, "noising")
        all_seeds_d = load_all_seeds(results_dirs, "denoising")
        n_seeds = len(all_seeds_n)
        
        # Find common epochs across both
        common_epochs = find_common_epochs(all_seeds_n + all_seeds_d)
        if not common_epochs:
            print("ERROR: No common epochs found across seeds. Exiting.")
            sys.exit(1)
        print(f"\n  Plotting {len(common_epochs)} common epochs: {common_epochs[0]} ... {common_epochs[-1]}")

        print("\nComputing mean and SEM across seeds...")
        means_n, _ = compute_mean_sem(all_seeds_n, common_epochs, metric)
        means_d, _ = compute_mean_sem(all_seeds_d, common_epochs, metric)

        components = get_components(all_seeds_n)
        attn_heads, mlps, embedding = parse_components(components)

        sharey = (args.sharey or cfg.get("sharey", "False")).lower() == "true"

        print("\n--- Generating attention head comparison plots ---")
        plot_attention_heads_comparison(common_epochs, means_n, means_d, metric, output_dir,
                                        attn_heads, smoothing_window, clip_percentile,
                                        experiment, hop, n_seeds, ylim_list=attn_ylim, sharey=sharey)

        print("\n--- Generating MLP & embedding comparison plot ---")
        plot_mlps_embedding_comparison(common_epochs, means_n, means_d, metric, output_dir,
                                       mlps, embedding, smoothing_window, clip_percentile,
                                       experiment, hop, n_seeds, ylim=mlp_ylim)
    else:
        # Load all seeds
        all_seeds = load_all_seeds(results_dirs, setup)
        n_seeds = len(all_seeds)

        # Find common epochs
        common_epochs = find_common_epochs(all_seeds)
        if not common_epochs:
            print("ERROR: No common epochs found across seeds. Exiting.")
            sys.exit(1)
        print(f"\n  Plotting {len(common_epochs)} common epochs: {common_epochs[0]} ... {common_epochs[-1]}")

        # Compute mean and SEM
        print("\nComputing mean and SEM across seeds...")
        means, sems = compute_mean_sem(all_seeds, common_epochs, metric)

        # Parse components
        components = get_components(all_seeds)
        attn_heads, mlps, embedding = parse_components(components)

        # Merge sharey from CLI / config
        sharey = (args.sharey or cfg.get("sharey", "False")).lower() == "true"

        # Plot
        print("\n--- Generating attention head line plots ---")
        plot_attention_heads(common_epochs, means, sems, setup, metric, output_dir,
                             attn_heads, smoothing_window, clip_percentile,
                             experiment, hop, n_seeds, ylim_list=attn_ylim, sharey=sharey)

        print("\n--- Generating MLP & embedding line plot ---")
        plot_mlps_embedding(common_epochs, means, sems, setup, metric, output_dir,
                            mlps, embedding, smoothing_window, clip_percentile,
                            experiment, hop, n_seeds, ylim=mlp_ylim)

    print("\nDone!")


if __name__ == "__main__":
    main()
