"""
linear_probe.py — Train binary linear probes on extracted representations
to predict the "in-support" property.

For each layer and each checkpoint epoch, a logistic regression probe is
trained (with cross-validation) on the extracted representations to predict:

    y = 1  if target_class_id ∈ support_stone_ids   (in-support)
    y = 0  otherwise                                  (out-of-support)

This reveals **at which layer and at which training epoch** the model
linearly encodes whether the answer is a support stone. Comparing probe
accuracies across equivalent vs non-equivalent hops provides
representation-level evidence for cross-hop transfer.

Usage:
    python src/models/linear_probe.py \
        --rep_dir representations_output_hop2/ \
        --output_dir probe_results/ \
        --n_folds 5
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
#  I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(rep_dir):
    """Load the in-support labels and target class IDs from labels.npz."""
    labels_path = os.path.join(rep_dir, "labels.npz")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"labels.npz not found in {rep_dir}. "
            "Run extract_representations.py first."
        )
    data = np.load(labels_path)
    return data["in_support_labels"], data["target_class_ids"]


def load_representation_file(fpath):
    """Load a single representations_epoch_*.npz file."""
    npz = np.load(fpath)
    epoch = int(npz["epoch"]) if "epoch" in npz.files else None
    layers = {}
    for key in npz.files:
        if key in ("epoch", "predicted_class_ids"):
            continue
        layers[key] = npz[key]
    predicted = npz["predicted_class_ids"] if "predicted_class_ids" in npz.files else None
    return epoch, layers, predicted


def discover_representation_files(rep_dir):
    """Find and sort all representation files by epoch."""
    files = sorted(glob.glob(os.path.join(rep_dir, "representations_epoch_*.npz")))
    regex = re.compile(r"representations_epoch_(\d+)\.npz")
    result = []
    for f in files:
        m = regex.search(os.path.basename(f))
        if m:
            result.append((int(m.group(1)), f))
    result.sort(key=lambda x: x[0])
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Probing
# ─────────────────────────────────────────────────────────────────────────────

def train_probe(X, y, n_folds=5, max_iter=1000, C=1.0):
    """
    Train a logistic regression probe with stratified K-fold cross-validation.

    Parameters
    ----------
    X : np.ndarray (N, d)
    y : np.ndarray (N,)  binary labels
    n_folds : int
    max_iter : int
    C : float  regularization strength (inverse)

    Returns
    -------
    dict with keys:
        accuracy_mean, accuracy_std,
        auroc_mean, auroc_std,
        fold_accuracies, fold_aurocs
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_accs = []
    fold_aurocs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train logistic regression
        clf = LogisticRegression(
            max_iter=max_iter,
            C=C,
            solver="lbfgs",
            random_state=42,
            class_weight="balanced",  # Handle class imbalance
        )
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        fold_accs.append(accuracy_score(y_test, y_pred))

        if len(np.unique(y_test)) > 1:
            y_prob = clf.predict_proba(X_test)[:, 1]
            fold_aurocs.append(roc_auc_score(y_test, y_prob))
        else:
            fold_aurocs.append(float("nan"))

    return {
        "accuracy_mean": np.mean(fold_accs),
        "accuracy_std": np.std(fold_accs),
        "auroc_mean": np.nanmean(fold_aurocs),
        "auroc_std": np.nanstd(fold_aurocs),
        "fold_accuracies": fold_accs,
        "fold_aurocs": fold_aurocs,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear probing on extracted representations."
    )
    parser.add_argument("--rep_dir", type=str, required=True,
                        help="Directory with representations_epoch_*.npz and labels.npz.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save probe results. Defaults to <rep_dir>/probe_results/.")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds for the probe.")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse regularization strength for logistic regression.")
    parser.add_argument("--max_iter", type=int, default=2000,
                        help="Max iterations for logistic regression.")
    parser.add_argument("--label", type=str, default="",
                        help="Optional label for plot titles (e.g., 'hop2_seed42').")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.rep_dir, "probe_results")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load labels ──────────────────────────────────────────────────────
    in_support_labels, target_class_ids = load_labels(args.rep_dir)
    print(f"Loaded {len(in_support_labels)} labels")
    print(f"  In-support: {in_support_labels.sum()} / {len(in_support_labels)} "
          f"({in_support_labels.mean():.4f})")

    # ── Discover representation files ────────────────────────────────────
    epoch_files = discover_representation_files(args.rep_dir)
    print(f"Found {len(epoch_files)} representation files")

    if len(epoch_files) == 0:
        print("No representation files found. Exiting.")
        return

    # ── Get layer names from first file ──────────────────────────────────
    _, first_layers, _ = load_representation_file(epoch_files[0][1])
    layer_names = sorted(first_layers.keys())
    print(f"Layers: {layer_names}")
    del first_layers

    # ── Probe each epoch × layer ─────────────────────────────────────────
    # Results: layer → { epochs: [...], accuracies: [...], aurocs: [...] }
    results = {layer: {"epochs": [], "acc_mean": [], "acc_std": [],
                        "auroc_mean": [], "auroc_std": []}
               for layer in layer_names}

    for epoch_num, fpath in tqdm(epoch_files, desc="Probing epochs"):
        epoch, layers, predicted = load_representation_file(fpath)
        if epoch is None:
            epoch = epoch_num

        for layer in layer_names:
            X = layers[layer].astype(np.float32)
            y = in_support_labels

            assert X.shape[0] == y.shape[0], (
                f"Mismatch: X has {X.shape[0]} examples, labels has {y.shape[0]}"
            )

            probe_result = train_probe(
                X, y,
                n_folds=args.n_folds,
                max_iter=args.max_iter,
                C=args.C,
            )

            results[layer]["epochs"].append(epoch)
            results[layer]["acc_mean"].append(probe_result["accuracy_mean"])
            results[layer]["acc_std"].append(probe_result["accuracy_std"])
            results[layer]["auroc_mean"].append(probe_result["auroc_mean"])
            results[layer]["auroc_std"].append(probe_result["auroc_std"])

        del layers
        # Free memory

    # ── Save raw results ─────────────────────────────────────────────────
    save_dict = {}
    for layer in layer_names:
        for metric in ("epochs", "acc_mean", "acc_std", "auroc_mean", "auroc_std"):
            save_dict[f"{layer}_{metric}"] = np.array(results[layer][metric])

    np.savez(os.path.join(args.output_dir, "probe_results.npz"), **save_dict)

    # ── Plot: Probe accuracy over epochs, one line per layer ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy plot
    ax = axes[0]
    for layer in layer_names:
        epochs = results[layer]["epochs"]
        acc = results[layer]["acc_mean"]
        acc_std = results[layer]["acc_std"]
        ax.plot(epochs, acc, label=layer, linewidth=1.5)
        ax.fill_between(epochs,
                        np.array(acc) - np.array(acc_std),
                        np.array(acc) + np.array(acc_std),
                        alpha=0.15)

    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Probe Accuracy (5-fold CV)")
    ax.set_title(f"In-Support Linear Probe — Accuracy{' (' + args.label + ')' if args.label else ''}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Chance level
    majority_class_rate = max(in_support_labels.mean(), 1 - in_support_labels.mean())
    ax.axhline(y=majority_class_rate, color="gray", linestyle="--",
               label=f"Majority baseline ({majority_class_rate:.3f})", alpha=0.7)
    ax.axhline(y=0.5, color="lightgray", linestyle=":", alpha=0.5)
    ax.legend()

    # AUROC plot
    ax = axes[1]
    for layer in layer_names:
        epochs = results[layer]["epochs"]
        auroc = results[layer]["auroc_mean"]
        auroc_std = results[layer]["auroc_std"]
        ax.plot(epochs, auroc, label=layer, linewidth=1.5)
        ax.fill_between(epochs,
                        np.array(auroc) - np.array(auroc_std),
                        np.array(auroc) + np.array(auroc_std),
                        alpha=0.15)

    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Probe AUROC (5-fold CV)")
    ax.set_title(f"In-Support Linear Probe — AUROC{' (' + args.label + ')' if args.label else ''}")
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle="--", label="Chance (0.5)", alpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "probe_accuracy_over_epochs.png"), dpi=150)
    plt.close(fig)

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Probe results saved to {args.output_dir}")
    print(f"{'='*60}")
    for layer in layer_names:
        final_acc = results[layer]["acc_mean"][-1]
        final_auroc = results[layer]["auroc_mean"][-1]
        best_acc = max(results[layer]["acc_mean"])
        best_epoch = results[layer]["epochs"][np.argmax(results[layer]["acc_mean"])]
        print(f"  {layer:12s} | Final acc: {final_acc:.4f} | "
              f"Best acc: {best_acc:.4f} (epoch {best_epoch}) | "
              f"Final AUROC: {final_auroc:.4f}")


if __name__ == "__main__":
    main()
