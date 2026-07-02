"""
temporal_cka.py — Compute CKA (Centered Kernel Alignment) across training
epochs and across hops, for every Transformer layer.

This script supports two analysis modes:

  1. **Temporal CKA (within one hop)**: For a single model's training run,
     compute CKA(layer_l at epoch_i, layer_l at epoch_j) for all pairs (i,j).
     Produces an epoch×epoch heatmap per layer showing how representations
     evolve during training.

  2. **Cross-hop CKA (between two hops at matched epochs)**: For two models
     trained on different hop counts, compute CKA(hop_A layer_l epoch_t,
     hop_B layer_l epoch_t) for every layer and epoch. Produces an epoch-wise
     CKA curve per layer.

CKA implementation follows Kornblith et al. (2019) — "Similarity of Neural
Network Representations Revisited" — using linear CKA (faster, closed-form).

Usage:
    # Temporal CKA for one model:
    python src/models/temporal_cka.py \
        --mode temporal \
        --rep_dir_a representations_output_hop2/ \
        --output_dir cka_results/

    # Cross-hop CKA:
    python src/models/temporal_cka.py \
        --mode cross_hop \
        --rep_dir_a representations_output_hop2/ \
        --rep_dir_b representations_output_hop4/ \
        --output_dir cka_results/
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
#  Linear CKA implementation
# ─────────────────────────────────────────────────────────────────────────────

def _center_gram(K):
    """Center a Gram matrix (in-place safe variant)."""
    n = K.shape[0]
    unit = np.ones((n, n), dtype=K.dtype) / n
    return K - unit @ K - K @ unit + unit @ K @ unit


def linear_cka(X, Y):
    """
    Compute linear CKA between two representation matrices.

    Parameters
    ----------
    X : np.ndarray, shape (n_examples, d_x)
    Y : np.ndarray, shape (n_examples, d_y)

    Returns
    -------
    float — CKA similarity in [0, 1].
    """
    assert X.shape[0] == Y.shape[0], (
        f"Number of examples must match: X has {X.shape[0]}, Y has {Y.shape[0]}"
    )
    n = X.shape[0]

    # Center columns
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # HSIC estimator using linear kernel: HSIC = ||Y^T X||_F^2 / (n-1)^2
    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    XtX = (X.T @ X)  # (d_x, d_x)
    YtY = (Y.T @ Y)  # (d_y, d_y)
    YtX = (Y.T @ X)  # (d_y, d_x)

    hsic_xy = np.sum(YtX ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def minibatch_cka(X, Y, batch_size=512):
    """
    Compute linear CKA in a memory-efficient way by processing in minibatches.
    Uses the equivalent formulation:
        CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    For very large N, we compute Y^T X in blocks.
    """
    n = X.shape[0]
    if n <= batch_size * 2:
        return linear_cka(X, Y)

    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # X^T X and Y^T Y can be computed in blocks
    XtX = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    YtY = np.zeros((Y.shape[1], Y.shape[1]), dtype=np.float64)
    YtX = np.zeros((Y.shape[1], X.shape[1]), dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb = X[start:end].astype(np.float64)
        Yb = Y[start:end].astype(np.float64)
        XtX += Xb.T @ Xb
        YtY += Yb.T @ Yb
        YtX += Yb.T @ Xb

    hsic_xy = np.sum(YtX ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


# ─────────────────────────────────────────────────────────────────────────────
#  I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_representation_epochs(rep_dir):
    """
    Load all representations_epoch_*.npz files from a directory.

    Returns
    -------
    dict[int, dict[str, np.ndarray]]
        epoch → { layer_name → (N, d_model) }
    """
    files = sorted(glob.glob(os.path.join(rep_dir, "representations_epoch_*.npz")))
    if not files:
        raise FileNotFoundError(f"No representation files in {rep_dir}")

    data = {}
    regex = re.compile(r"representations_epoch_(\d+)\.npz")
    for fpath in files:
        m = regex.search(os.path.basename(fpath))
        if not m:
            continue
        epoch = int(m.group(1))
        npz = np.load(fpath)
        layers = {}
        for key in npz.files:
            if key in ("epoch", "predicted_class_ids"):
                continue
            layers[key] = npz[key]
        data[epoch] = layers

    print(f"Loaded {len(data)} epochs from {rep_dir}")
    print(f"Layer names: {list(next(iter(data.values())).keys())}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  Temporal CKA: epoch × epoch heatmap per layer
# ─────────────────────────────────────────────────────────────────────────────

def compute_temporal_cka(rep_data, output_dir, label=""):
    """
    Compute CKA(layer_l at epoch_i, layer_l at epoch_j) for all (i,j) pairs
    and for every layer. Save heatmaps and raw matrices.
    """
    epochs = sorted(rep_data.keys())
    layer_names = sorted(rep_data[epochs[0]].keys())
    n_epochs = len(epochs)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nComputing temporal CKA for {n_epochs} epochs × {len(layer_names)} layers")

    for layer in layer_names:
        print(f"  Layer: {layer}")
        cka_matrix = np.zeros((n_epochs, n_epochs), dtype=np.float32)

        for i in tqdm(range(n_epochs), desc=f"  {layer}", leave=False):
            Xi = rep_data[epochs[i]][layer].astype(np.float32)
            # Diagonal is always 1
            cka_matrix[i, i] = 1.0
            for j in range(i + 1, n_epochs):
                Xj = rep_data[epochs[j]][layer].astype(np.float32)
                cka_val = minibatch_cka(Xi, Xj)
                cka_matrix[i, j] = cka_val
                cka_matrix[j, i] = cka_val

        # Save raw matrix
        np.save(os.path.join(output_dir, f"temporal_cka_{layer}{label}.npy"), cka_matrix)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cka_matrix, cmap="magma", vmin=0, vmax=1,
                       origin="lower", aspect="auto")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Epoch")
        ax.set_title(f"Temporal CKA — {layer}{label}")

        # Tick labels
        tick_step = max(1, n_epochs // 10)
        tick_positions = list(range(0, n_epochs, tick_step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(epochs[t]) for t in tick_positions], rotation=45)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([str(epochs[t]) for t in tick_positions])

        plt.colorbar(im, ax=ax, label="CKA")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"temporal_cka_{layer}{label}.png"), dpi=150)
        plt.close(fig)

    print(f"  Saved temporal CKA matrices and heatmaps to {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-hop CKA: compare two models at matched epochs, per layer
# ─────────────────────────────────────────────────────────────────────────────

def compute_cross_hop_cka(rep_data_a, rep_data_b, output_dir,
                          label_a="hop_A", label_b="hop_B",
                          subsample_n=None):
    """
    For matched epochs, compute CKA(hop_A layer_l, hop_B layer_l).

    Because the two models are evaluated on different data (different hop
    counts), the number of examples N may differ. We handle this by
    subsampling to the minimum N, or a user-specified subsample_n.

    NOTE: For cross-hop CKA to be meaningful, both models should be
    evaluated on the **same** validation set (e.g., both on the 2-hop val
    data). This way N is the same and examples are aligned.
    """
    common_epochs = sorted(set(rep_data_a.keys()) & set(rep_data_b.keys()))
    if not common_epochs:
        raise ValueError("No common epochs between the two representation directories.")

    layer_names = sorted(rep_data_a[common_epochs[0]].keys())
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nCross-hop CKA: {label_a} vs {label_b}")
    print(f"  Common epochs: {len(common_epochs)}")

    # Storage: layer → list of CKA values (one per epoch)
    cka_curves = {layer: [] for layer in layer_names}

    for epoch in tqdm(common_epochs, desc="Cross-hop CKA"):
        for layer in layer_names:
            Xa = rep_data_a[epoch][layer].astype(np.float32)
            Xb = rep_data_b[epoch][layer].astype(np.float32)

            # Handle different N by subsampling
            n_a, n_b = Xa.shape[0], Xb.shape[0]
            n = min(n_a, n_b)
            if subsample_n is not None:
                n = min(n, subsample_n)
            if n < n_a:
                Xa = Xa[:n]
            if n < n_b:
                Xb = Xb[:n]

            cka_val = minibatch_cka(Xa, Xb)
            cka_curves[layer].append(cka_val)

    # Save raw curves
    np.savez(
        os.path.join(output_dir, f"cross_hop_cka_{label_a}_vs_{label_b}.npz"),
        epochs=np.array(common_epochs),
        **{layer: np.array(vals) for layer, vals in cka_curves.items()},
    )

    # Plot: one line per layer
    fig, ax = plt.subplots(figsize=(12, 6))
    for layer in layer_names:
        ax.plot(common_epochs, cka_curves[layer], label=layer, linewidth=1.5)

    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("CKA Similarity")
    ax.set_title(f"Cross-Hop CKA: {label_a} vs {label_b}")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"cross_hop_cka_{label_a}_vs_{label_b}.png"),
        dpi=150,
    )
    plt.close(fig)

    print(f"  Saved cross-hop CKA to {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-hop Temporal CKA: full epoch_A × epoch_B heatmap per layer
# ─────────────────────────────────────────────────────────────────────────────

def compute_cross_hop_temporal_cka(rep_data_a, rep_data_b, output_dir,
                                   label_a="hop_A", label_b="hop_B"):
    """
    Compute full CKA(hop_A at epoch_i, hop_B at epoch_j) for ALL (i,j) pairs,
    per layer. This produces epoch_A × epoch_B heatmaps.

    This is the most informative visualization: it shows not only whether
    representations are similar at the same epoch, but whether hop_A at some
    early epoch matches hop_B at a later epoch (or vice versa).
    """
    epochs_a = sorted(rep_data_a.keys())
    epochs_b = sorted(rep_data_b.keys())
    layer_names = sorted(rep_data_a[epochs_a[0]].keys())
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nCross-hop temporal CKA: {label_a} ({len(epochs_a)} epochs) "
          f"vs {label_b} ({len(epochs_b)} epochs)")

    for layer in layer_names:
        print(f"  Layer: {layer}")
        cka_matrix = np.zeros((len(epochs_a), len(epochs_b)), dtype=np.float32)

        for i in tqdm(range(len(epochs_a)), desc=f"  {layer}", leave=False):
            Xi = rep_data_a[epochs_a[i]][layer].astype(np.float32)
            for j in range(len(epochs_b)):
                Xj = rep_data_b[epochs_b[j]][layer].astype(np.float32)
                n = min(Xi.shape[0], Xj.shape[0])
                cka_matrix[i, j] = minibatch_cka(Xi[:n], Xj[:n])

        # Save
        np.savez(
            os.path.join(output_dir, f"cross_temporal_cka_{layer}_{label_a}_vs_{label_b}.npz"),
            cka_matrix=cka_matrix,
            epochs_a=np.array(epochs_a),
            epochs_b=np.array(epochs_b),
        )

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cka_matrix, cmap="magma", vmin=0, vmax=1,
                       origin="lower", aspect="auto")
        ax.set_xlabel(f"Epoch ({label_b})")
        ax.set_ylabel(f"Epoch ({label_a})")
        ax.set_title(f"Cross-Hop Temporal CKA — {layer}\n{label_a} vs {label_b}")

        tick_step_a = max(1, len(epochs_a) // 10)
        tick_step_b = max(1, len(epochs_b) // 10)
        ax.set_yticks(range(0, len(epochs_a), tick_step_a))
        ax.set_yticklabels([str(epochs_a[t]) for t in range(0, len(epochs_a), tick_step_a)])
        ax.set_xticks(range(0, len(epochs_b), tick_step_b))
        ax.set_xticklabels([str(epochs_b[t]) for t in range(0, len(epochs_b), tick_step_b)], rotation=45)

        plt.colorbar(im, ax=ax, label="CKA")
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"cross_temporal_cka_{layer}_{label_a}_vs_{label_b}.png"),
            dpi=150,
        )
        plt.close(fig)

    print(f"  Saved cross-hop temporal CKA to {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CKA analysis on extracted representations.")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["temporal", "cross_hop", "cross_temporal", "all"],
                        help="Analysis mode.")
    parser.add_argument("--rep_dir_a", type=str, required=True,
                        help="Directory with representations_epoch_*.npz files (model A).")
    parser.add_argument("--rep_dir_b", type=str, default=None,
                        help="Directory with representations for model B "
                             "(required for cross_hop and cross_temporal modes).")
    parser.add_argument("--label_a", type=str, default="model_A",
                        help="Label for model A in plots.")
    parser.add_argument("--label_b", type=str, default="model_B",
                        help="Label for model B in plots.")
    parser.add_argument("--output_dir", type=str, default="cka_results",
                        help="Where to save CKA results.")
    parser.add_argument("--subsample_n", type=int, default=None,
                        help="Subsample to this many examples for speed.")

    return parser.parse_args()


def main():
    args = parse_args()

    rep_data_a = load_representation_epochs(args.rep_dir_a)

    if args.mode == "temporal":
        compute_temporal_cka(rep_data_a, args.output_dir, label=f" ({args.label_a})")

    elif args.mode == "cross_hop":
        if args.rep_dir_b is None:
            raise ValueError("--rep_dir_b required for cross_hop mode")
        rep_data_b = load_representation_epochs(args.rep_dir_b)
        compute_cross_hop_cka(
            rep_data_a, rep_data_b, args.output_dir,
            label_a=args.label_a, label_b=args.label_b,
            subsample_n=args.subsample_n,
        )

    elif args.mode == "cross_temporal":
        if args.rep_dir_b is None:
            raise ValueError("--rep_dir_b required for cross_temporal mode")
        rep_data_b = load_representation_epochs(args.rep_dir_b)
        compute_cross_hop_temporal_cka(
            rep_data_a, rep_data_b, args.output_dir,
            label_a=args.label_a, label_b=args.label_b,
        )

    elif args.mode == "all":
        # Run all analyses
        compute_temporal_cka(rep_data_a, args.output_dir, label=f" ({args.label_a})")

        if args.rep_dir_b is not None:
            rep_data_b = load_representation_epochs(args.rep_dir_b)
            compute_cross_hop_cka(
                rep_data_a, rep_data_b, args.output_dir,
                label_a=args.label_a, label_b=args.label_b,
                subsample_n=args.subsample_n,
            )
            compute_cross_hop_temporal_cka(
                rep_data_a, rep_data_b, args.output_dir,
                label_a=args.label_a, label_b=args.label_b,
            )


if __name__ == "__main__":
    main()
