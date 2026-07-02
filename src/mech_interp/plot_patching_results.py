"""
plot_patching_results.py

Visualizes activation patching results from epoch-range pickle files produced by
chemistry_identity_pairs.py sweeps.

Produces:
  - Line plots: epoch vs lse_mean score per component group (attention heads / MLPs)
  - Heatmap PNG: layer x head grid at a specific epoch (--heatmap_epoch)
  - Heatmap GIF: animated heatmap sweeping over all epochs

Usage examples:
  # Line plots (noising)
  python src/models/mech_interp_exps/plot_patching_results.py \\
      --results_dir /path/to/init_seed_3 \\
      --setup noising \\
      --output_dir /path/to/plots

  # Heatmap at a specific epoch
  python src/models/mech_interp_exps/plot_patching_results.py \\
      --results_dir /path/to/init_seed_3 \\
      --setup noising \\
      --output_dir /path/to/plots \\
      --heatmap_epoch 500

  # All line plots + animated heatmap GIF
  python src/models/mech_interp_exps/plot_patching_results.py \\
      --results_dir /path/to/init_seed_3 \\
      --setup noising \\
      --output_dir /path/to/plots \\
      --animate
"""

import argparse
import glob
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for clusters)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_dir: str, setup: str) -> dict:
    """
    Load and merge all epoch-range pickle files for the given setup
    (noising or denoising) from results_dir.

    Only the scalar *_mean values are retained (softmax_mean, lse_mean,
    raw_lse_mean). The per-pair arrays (softmax_all, lse_all, raw_lse_all)
    are discarded immediately to avoid OOM on machines with limited RAM.

    Returns a dict keyed by epoch (int) -> component_name -> metric_dict.
    """
    SCALAR_KEYS = {"softmax_mean", "lse_mean", "raw_lse_mean"}

    pattern = os.path.join(results_dir, f"layer_head_sweep_results_epochs_*_{setup}.pkl")
    pkl_files = sorted(glob.glob(pattern))

    if not pkl_files:
        raise FileNotFoundError(
            f"No pickle files found matching '{pattern}'.\n"
            f"Check --results_dir and --setup arguments."
        )

    merged: dict = {}
    for i, path in enumerate(pkl_files):
        print(f"  [{i+1}/{len(pkl_files)}] Loading {os.path.basename(path)} ...", flush=True)
        with open(path, "rb") as f:
            data = pickle.load(f)
        for epoch, comp_dict in data.items():
            if epoch not in merged:
                # Keep only scalar means -- drop large per-pair arrays
                merged[epoch] = {
                    comp: {k: v for k, v in metrics.items() if k in SCALAR_KEYS}
                    for comp, metrics in comp_dict.items()
                }
            # If epoch already exists (overlapping ranges), keep existing
    return merged


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------

def parse_components(merged: dict):
    """
    Return sorted list of all component names plus grouped lists.
    """
    sample_epoch = next(iter(merged))
    all_comps = list(merged[sample_epoch].keys())

    attn_heads = sorted(
        [c for c in all_comps if re.match(r"layer_\d+_head_\d+", c)],
        key=lambda c: (int(re.search(r"layer_(\d+)", c).group(1)),
                       int(re.search(r"head_(\d+)", c).group(1)))
    )
    mlps = sorted(
        [c for c in all_comps if c.endswith("_mlp_out")],
        key=lambda c: int(re.search(r"layer_(\d+)", c).group(1))
    )
    embedding = [c for c in all_comps if c == "embedding"]
    return all_comps, attn_heads, mlps, embedding


def get_layer_head_matrix(merged: dict, epoch: int, metric: str = "lse_mean"):
    """
    Build a 2D numpy array (n_layers x n_heads) for a specific epoch.
    """
    epoch_data = merged[epoch]
    # Discover shape
    layer_ids = sorted(set(
        int(re.search(r"layer_(\d+)", c).group(1))
        for c in epoch_data if re.match(r"layer_\d+_head_\d+", c)
    ))
    head_ids = sorted(set(
        int(re.search(r"head_(\d+)", c).group(1))
        for c in epoch_data if re.match(r"layer_\d+_head_\d+", c)
    ))
    n_layers = len(layer_ids)
    n_heads = len(head_ids)
    mat = np.zeros((n_layers, n_heads))
    for li, layer in enumerate(layer_ids):
        for hi, head in enumerate(head_ids):
            key = f"layer_{layer}_head_{head}"
            mat[li, hi] = epoch_data.get(key, {}).get(metric, 0.0)
    return mat, layer_ids, head_ids


# ---------------------------------------------------------------------------
# Plot 1: Line plots
# ---------------------------------------------------------------------------

def plot_line(merged: dict, setup: str, output_dir: str, metric: str = "lse_mean",
              clip_percentile: float = 99.0, smoothing_window: int = 25,
              exp_type: str = "unknown_exp", init_seed: str = "unknown_seed"):
    """
    Produces two line-plot figures:
      1. Attention heads (one line per head, faceted by layer)
      2. MLPs + embedding

    clip_percentile: clamp the y-axis to this percentile of absolute values
      across all components, so that rare outlier spikes don't crush the
      visible signal. Set to 100 to disable clipping.
    smoothing_window: rolling-average window (in epochs) for the trend line.
      Raw scores are plotted faintly behind the smoothed line.
      Set to 1 to disable smoothing.
    """
    _, attn_heads, mlps, embedding = parse_components(merged)
    epochs = sorted(merged.keys())

    def _scores(comp_list):
        return {
            c: [merged[e].get(c, {}).get(metric, float("nan")) for e in epochs]
            for c in comp_list
        }

    def _smooth(vals):
        """Apply a simple rolling average over smoothing_window steps."""
        if smoothing_window <= 1:
            return vals
        arr = np.array(vals, dtype=float)
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = np.convolve(arr, kernel, mode="same")
        # Edges are biased by zero-padding from convolve; restore as NaN
        half = smoothing_window // 2
        smoothed[:half] = np.nan
        smoothed[-half:] = np.nan
        return smoothed.tolist()

    def _ylim(scores_dict):
        """Compute symmetric y-limits clipped at clip_percentile."""
        all_vals = [v for vals in scores_dict.values() for v in vals
                    if not (v != v)]  # exclude NaN
        if not all_vals:
            return None
        limit = float(np.nanpercentile(np.abs(all_vals), clip_percentile))
        return (-limit, limit)

    def _draw_lines(ax, comp_list, scores, cmap, label_fn):
        for i, comp in enumerate(comp_list):
            color = cmap(i)
            raw = scores[comp]
            smoothed = _smooth(raw)
            # Raw: faint background
            ax.plot(epochs, raw, color=color, linewidth=0.6, alpha=0.25)
            # Smoothed: bold foreground
            ax.plot(epochs, smoothed, color=color, linewidth=1.8,
                    label=label_fn(comp))

    # ---- Attention heads ------------------------------------------------
    layer_ids = sorted(set(
        int(re.search(r"layer_(\d+)", c).group(1)) for c in attn_heads
    ))
    n_layers = len(layer_ids)
    head_scores = _scores(attn_heads)
    ylim_attn = _ylim(head_scores)

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4), sharey=True)
    if n_layers == 1:
        axes = [axes]
    fig.suptitle(f"Attention Head Patching Scores ({setup}) — {metric}", fontsize=13)

    cmap = plt.get_cmap("tab10")
    for ax, layer in zip(axes, layer_ids):
        layer_comps = [c for c in attn_heads if f"layer_{layer}_" in c]
        def _attn_label(c, _re=re):
            return "head " + _re.search(r'head_(\d+)', c).group(1)
        _draw_lines(ax, layer_comps, head_scores, cmap, label_fn=_attn_label)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Epoch")
        if layer == layer_ids[0]:
            ax.set_ylabel(metric)
        if ylim_attn:
            ax.set_ylim(ylim_attn)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"line_attn_heads_{exp_type}_{init_seed}_{setup}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    # ---- MLPs + embedding -----------------------------------------------
    mlp_emb = embedding + mlps
    mlp_scores = _scores(mlp_emb)
    ylim_mlp = _ylim(mlp_scores)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(f"MLP & Embedding Patching Scores ({setup}) — {metric}", fontsize=13)
    cmap2 = plt.get_cmap("tab10")
    _draw_lines(ax, mlp_emb, mlp_scores, cmap2, label_fn=lambda c: c)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    if ylim_mlp:
        ax.set_ylim(ylim_mlp)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"line_mlp_embedding_{exp_type}_{init_seed}_{setup}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Static heatmap at a specific epoch
# ---------------------------------------------------------------------------

def plot_heatmap(merged: dict, epoch: int, setup: str, output_dir: str,
                 metric: str = "lse_mean", exp_type: str = "unknown_exp", init_seed: str = "unknown_seed"):
    """
    Save a layer x head heatmap for the given epoch.
    """
    if epoch not in merged:
        available = sorted(merged.keys())
        raise ValueError(
            f"Epoch {epoch} not found in data. Available: {available[:10]}..."
        )
    mat, layer_ids, head_ids = get_layer_head_matrix(merged, epoch, metric)

    fig, ax = plt.subplots(figsize=(max(4, len(head_ids) * 1.2), max(3, len(layer_ids) * 0.8)))
    vmax = max(abs(mat.max()), abs(mat.min()))
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(head_ids)))
    ax.set_xticklabels([f"Head {h}" for h in head_ids])
    ax.set_yticks(range(len(layer_ids)))
    ax.set_yticklabels([f"Layer {l}" for l in layer_ids])
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"Patching Score Heatmap — Epoch {epoch} ({setup}) — {metric}")
    plt.colorbar(im, ax=ax, label=metric)
    # Annotate cells with values
    for li in range(len(layer_ids)):
        for hi in range(len(head_ids)):
            ax.text(hi, li, f"{mat[li, hi]:.3f}", ha="center", va="center",
                    fontsize=7, color="black")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"heatmap_epoch_{epoch}_{exp_type}_{init_seed}_{setup}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Animated heatmap GIF
# ---------------------------------------------------------------------------

def plot_animated_heatmap(merged: dict, setup: str, output_dir: str,
                          metric: str = "lse_mean", fps: int = 5,
                          exp_type: str = "unknown_exp", init_seed: str = "unknown_seed"):
    """
    Produce an animated GIF sweeping through all epochs.
    """
    epochs = sorted(merged.keys())
    # Pre-compute global colour scale
    all_vals = []
    for ep in epochs:
        mat, layer_ids, head_ids = get_layer_head_matrix(merged, ep, metric)
        all_vals.extend(mat.flatten().tolist())
    vmax = max(abs(np.nanmax(all_vals)), abs(np.nanmin(all_vals)))
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(max(4, len(head_ids) * 1.2), max(3, len(layer_ids) * 0.8)))
    mat0, _, _ = get_layer_head_matrix(merged, epochs[0], metric)
    im = ax.imshow(mat0, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(head_ids)))
    ax.set_xticklabels([f"Head {h}" for h in head_ids])
    ax.set_yticks(range(len(layer_ids)))
    ax.set_yticklabels([f"Layer {l}" for l in layer_ids])
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    title = ax.set_title(f"Patching Heatmap — Epoch {epochs[0]} ({setup}) — {metric}")
    plt.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()

    texts = []
    for li in range(len(layer_ids)):
        row = []
        for hi in range(len(head_ids)):
            t = ax.text(hi, li, "", ha="center", va="center", fontsize=7, color="black")
            row.append(t)
        texts.append(row)

    def update(frame_idx):
        ep = epochs[frame_idx]
        mat, _, _ = get_layer_head_matrix(merged, ep, metric)
        im.set_data(mat)
        title.set_text(f"Patching Heatmap — Epoch {ep} ({setup}) — {metric}")
        for li in range(len(layer_ids)):
            for hi in range(len(head_ids)):
                texts[li][hi].set_text(f"{mat[li, hi]:.3f}")
        return [im, title] + [t for row in texts for t in row]

    interval_ms = int(1000 / fps)
    ani = animation.FuncAnimation(
        fig, update, frames=len(epochs), interval=interval_ms, blit=False
    )

    out_path = os.path.join(output_dir, f"heatmap_animated_{exp_type}_{init_seed}_{setup}.gif")
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_run_metadata(results_dir: str):
    """
    Extracts exp_type and init_seed from the results_dir path.
    """
    parts = os.path.normpath(results_dir).split(os.sep)
    exp_type = "unknown_exp"
    init_seed = "unknown_seed"
    
    for part in reversed(parts):
        if "init_seed_" in part or part.startswith("init_seed"):
            init_seed = part
            break
    else:
        for part in reversed(parts):
            if part.isdigit():
                init_seed = f"seed_{part}"
                break
                
    for i, part in enumerate(parts):
        if part in ("mech_interp_results", "saved_models") and i + 1 < len(parts):
            exp_type = parts[i+1]
            break
            
    return exp_type, init_seed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize activation patching sweep results from pickle files."
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Directory containing epoch-range pickle files (e.g. layer_head_sweep_results_epochs_*_noising.pkl)."
    )
    parser.add_argument(
        "--setup", type=str, default="noising", choices=["noising", "denoising"],
        help="Experiment setup to visualize. Default: noising."
    )
    parser.add_argument(
        "--metric", type=str, default="raw_lse_mean",
        choices=["lse_mean", "softmax_mean", "raw_lse_mean"],
        help="Score metric to plot. Default: lse_mean."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save output plots."
    )
    parser.add_argument(
        "--heatmap_epoch", type=int, default=None,
        help="If set, save a static heatmap PNG for this specific epoch."
    )
    parser.add_argument(
        "--animate", type=str, default="False", choices=["True", "False"],
        help="If True, produce an animated GIF heatmap sweeping all epochs. Default: False."
    )
    parser.add_argument(
        "--fps", type=int, default=5,
        help="Frames per second for the animated heatmap GIF. Default: 5."
    )
    parser.add_argument(
        "--line_plots", type=str, default="True", choices=["True", "False"],
        help="If True, produce epoch vs score line plots. Default: True."
    )
    parser.add_argument(
        "--clip_percentile", type=float, default=100.0,
        help="Clamp line-plot y-axis to this percentile of absolute values to suppress outlier spikes. Set to 100 to disable. Default: 99."
    )
    parser.add_argument(
        "--smoothing_window", type=int, default=25,
        help="Rolling-average window (epochs) for the trend line in line plots. Raw scores shown faintly behind. Set to 1 to disable. Default: 25."
    )

    args = parser.parse_args()
    args.animate = str(args.animate).lower() == "true"
    args.line_plots = str(args.line_plots).lower() == "true"

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    print(f"Setup: {args.setup}  |  Metric: {args.metric}")
    merged = load_all_results(args.results_dir, args.setup)
    epochs = sorted(merged.keys())
    print(f"Loaded {len(epochs)} epochs: {epochs[0]} … {epochs[-1]}")

    exp_type, init_seed = extract_run_metadata(args.results_dir)

    if args.line_plots:
        print("\n--- Generating line plots ---")
        plot_line(merged, args.setup, args.output_dir, metric=args.metric,
                  clip_percentile=args.clip_percentile,
                  smoothing_window=args.smoothing_window,
                  exp_type=exp_type, init_seed=init_seed)

    if args.heatmap_epoch is not None:
        print(f"\n--- Generating static heatmap for epoch {args.heatmap_epoch} ---")
        plot_heatmap(merged, args.heatmap_epoch, args.setup, args.output_dir, metric=args.metric,
                     exp_type=exp_type, init_seed=init_seed)

    if args.animate:
        print("\n--- Generating animated heatmap GIF ---")
        plot_animated_heatmap(merged, args.setup, args.output_dir, metric=args.metric, fps=args.fps,
                              exp_type=exp_type, init_seed=init_seed)

    if not args.line_plots and args.heatmap_epoch is None and not args.animate:
        print("Nothing to do. Use --line_plots True, --heatmap_epoch <epoch>, or --animate True.")


if __name__ == "__main__":
    main()
