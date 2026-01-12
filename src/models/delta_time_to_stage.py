"""
delta_time_to_stage.py

Minimal delta-t (delay) computation for event curves saved as NumPy .npz.

Each .npz must contain:
  - epochs: 1D array (E,)
  - one or more metric arrays, each 1D (E,)

This script computes:
  delta_t = t_intervention - t_baseline

where t is the first epoch where the chosen metric >= tau for `consecutive`
consecutive evaluation points.

Usage
-----
# Compute delta-t for a chosen metric key
python src/models/delta_time_to_stage.py \
  --baseline_npz /abs/path/to/baseline_metrics.npz \
  --intervention_npz /abs/path/to/freeze_layer0_metrics.npz \
  --metric_key EVENT_B_GIVEN_A \
  --tau 0.90 --consecutive 3

# List available keys in a .npz
python src/models/delta_time_to_stage.py \
  --baseline_npz /abs/path/to/baseline_metrics.npz \
  --list_keys
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pickle
import numpy as np


from baseline_and_frozen_filepaths import held_out_file_paths, frozen_held_out_file_paths_per_layer_per_init_seed, \
                staged_accuracies_held_out_file_paths_baseline_pickles, staged_accuracies_held_out_file_paths_frozen_pickles


# -----------------------------
# Placeholder metric names
# -----------------------------
# Edit these later to match what you save from analyze_predictions.py.
# You can also ignore these and pass a raw .npz key name via --metric_key.
METRIC_KEYS: Dict[str, str] = {
    "EVENT_A": "predicted_in_context_accuracies",
    "EVENT_B_GIVEN_A": "predicted_in_context_correct_half_accuracies",
    "EVENT_C_GIVEN_AB": "predicted_in_context_correct_half_exact_accuracies" 
}


@dataclass(frozen=True)
class StageSpec:
    tau: float = 0.90  # threshold for considering a stage is 'complete'. This is just the accuracy value.
    consecutive: int = 3  # number of consecutive points to consider a stage 'complete'. This means that the metric must be >= tau for `consecutive` eval points.


def load_npz(path: str) -> np.lib.npyio.NpzFile:
    if not path.lower().endswith(".npz"):
        raise ValueError(f"Expected a .npz file, got: {path}")
    return np.load(path, allow_pickle=False)


def list_keys(npz: np.lib.npyio.NpzFile) -> Tuple[str, ...]:
    return tuple(npz.files)


def first_sustained_crossing(
    epochs: np.ndarray, values: np.ndarray, spec: StageSpec
) -> Optional[int]:
    """First epoch where values >= tau for `consecutive` consecutive eval points."""
    if epochs.ndim != 1 or values.ndim != 1:
        raise ValueError("epochs and values must be 1D arrays")
    if epochs.shape[0] != values.shape[0]:
        raise ValueError("epochs and values must have same length")
    if spec.consecutive <= 0:
        raise ValueError("consecutive must be >= 1")

    c = 0
    for i, v in enumerate(values.tolist()):
        if float(v) >= spec.tau:
            c += 1
            if c >= spec.consecutive:
                start_idx = i - spec.consecutive + 1
                return int(epochs[start_idx])
        else:
            c = 0
    return None



def load_stagewise_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    # epochs could be range; normalize to np.array
    epochs = np.array(list(payload["epochs"]), dtype=int)
    return {"epochs": epochs, "seed_results": payload["seed_results"]}

def get_curve(payload: dict, seed: int, metric_key: str) -> np.ndarray:
    sr = payload["seed_results"]
    if seed not in sr:
        raise KeyError(f"Seed {seed} not found. Available: {sorted(sr.keys())}")
    if metric_key not in sr[seed]:
        raise KeyError(f"Metric '{metric_key}' not found for seed {seed}. Available: {list(sr[seed].keys())}")
    return np.asarray(sr[seed][metric_key], dtype=float)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute delta-t (delay) between baseline and intervention from .npz curves"
    )

    p.add_argument(
        "--baseline_pickle",
        type=str,
        required=False,
        help="Absolute path to baseline .npz (must contain 'epochs' and metric arrays).",
    )
    p.add_argument(
        "--intervention_pickle",
        type=str,
        default=False,
        help="Absolute path to intervention .npz. Required unless --list_keys is used.",
    )

    p.add_argument(
        "--metric_key",
        type=str,
        default="EVENT_B_GIVEN_A",
        help=(
            "Metric key to use for stage time. You may pass either a key in METRIC_KEYS "
            "(e.g., EVENT_C_GIVEN_AB) or a raw .npz key name (e.g., p_b_given_a)."
        ),
    )

    p.add_argument("--tau", type=float, default=0.90)
    p.add_argument("--consecutive", type=int, default=3)

    p.add_argument(
        "--list_keys",
        action="store_true",
        help="If set, list keys in --baseline_npz and exit.",
    )

    p.add_argument('--seed', type=int, default=0, help='Seed index to use from the .npz files.')

    return p.parse_args()


def main() -> None:
    args = parse_args()

    args.baseline_pickle = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/stagewise_accuracies_data_split_seed_0_init_seed_42_hop_4_exp_held_out.pkl"

    args.intervention_pickle = '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/stagewise_accuracies_frozen_layer_transformer_layer_1_freeze_epoch_100_data_split_seed_0_init_seed_42_hop_4_exp_held_out.pkl'

    base_payload = load_stagewise_pickle(args.baseline_pickle)

    if args.list_keys:
        print(
            json.dumps(
                {
                    "file": args.baseline_pickle,
                    "keys": list(list_keys(base_payload)),
                    "placeholder_keys": METRIC_KEYS,
                },
                indent=2,
            )
        )
        return

    if not args.intervention_pickle:
        raise ValueError("--intervention_pickle is required unless --list_keys is set")

    int_payload = load_stagewise_pickle(args.intervention_pickle)

    base_epochs = base_payload["epochs"]
    int_epochs = int_payload["epochs"]


    # Resolve metric key: allow either placeholder name or raw key.
    metric_key = METRIC_KEYS.get(args.metric_key, args.metric_key)

    base_values = get_curve(base_payload, seed=0, metric_key=metric_key)
    int_values = get_curve(int_payload, seed=0, metric_key=metric_key)
    # Validate presence
    if "epochs" not in base_payload:
        raise KeyError(f"baseline npz missing required key 'epochs'. Keys: {base_payload}")
    if "epochs" not in int_payload:
        raise KeyError(f"intervention npz missing required key 'epochs'. Keys: {int_payload}")
    

    if metric_key not in base_payload['seed_results'][args.seed]:
        raise KeyError(f"baseline npz missing metric '{metric_key}'. Keys: {base_payload['seed_results'][args.seed].keys()}")
    if metric_key not in int_payload['seed_results'][args.seed]:
        raise KeyError(f"intervention npz missing metric '{metric_key}'. Keys: {int_payload['seed_results'][args.seed].keys()}")

    base_epochs = np.asarray(base_payload["epochs"]).astype(int)[:-1]
    int_epochs = np.asarray(int_payload["epochs"]).astype(int)[:-1]

    base_values = np.asarray(base_payload['seed_results'][args.seed][metric_key]).astype(float)
    int_values = np.asarray(int_payload['seed_results'][args.seed][metric_key]).astype(float)

    spec = StageSpec(tau=float(args.tau), consecutive=int(args.consecutive))
    import pdb; pdb.set_trace()

    t_base = first_sustained_crossing(base_epochs, base_values, spec)
    interv_first_epoch = int(np.min(int_epochs))
    freeze_epoch = interv_first_epoch - 1  # by your guarantee: first saved = freeze + 1
    import pdb; pdb.set_trace()

    if len(int_epochs) > 0:
        assert int_epochs.min() == freeze_epoch + 1, (
            f"Intervention epochs should start at freeze_epoch + 1 = {freeze_epoch + 1}, "
            f"but got min epoch = {int_epochs.min()}"
        )

    # If baseline reached the stage before (or at) the freeze point, the intervention shares that
    # entire history, so the stage time is identical.
    if t_base is not None and int(t_base) <= int(freeze_epoch):
        t_int = t_base
        print(f"Intervention shares baseline history up to epoch {freeze_epoch}, so t_intervention = t_baseline = {t_base}")
    else:
        # Hybrid curve: last (consecutive-1) baseline points before intervention + intervention suffix.
        prefix_needed = max(spec.consecutive - 1, 0)
        prefix_mask = base_epochs < interv_first_epoch
        base_prefix_epochs = base_epochs[prefix_mask][-prefix_needed:] if prefix_needed > 0 and prefix_mask.any() else np.array([], dtype=int)

        if base_prefix_epochs.size > 0:
            base_map = {int(e): float(v) for e, v in zip(base_epochs.tolist(), base_values.tolist())}
            prefix_vals = np.array([base_map[int(e)] for e in base_prefix_epochs.tolist()], dtype=float)
        else:
            prefix_vals = np.array([], dtype=float)

        hybrid_epochs = np.concatenate([base_prefix_epochs, int_epochs]).astype(int)
        hybrid_values = np.concatenate([prefix_vals, int_values]).astype(float)

        t_int = first_sustained_crossing(hybrid_epochs, hybrid_values, spec)

    if t_base is None:
        raise RuntimeError(
            f"Baseline never reached stage for metric={metric_key} with tau={spec.tau}, consecutive={spec.consecutive}."
        )

    if t_int is None:
        out = {
            "metric_key": metric_key,
            "t_baseline": int(t_base),
            "t_intervention": None,
            "delta_t": float("inf"),
            "tau": spec.tau,
            "consecutive": spec.consecutive,
            "note": "intervention never reached stage within available epochs",
        }
        print(json.dumps(out, indent=2))
        return

    out = {
        "metric_key": metric_key,
        "t_baseline": int(t_base),
        "t_intervention": int(t_int),
        "delta_t": int(t_int - t_base),
        "tau": spec.tau,
        "consecutive": spec.consecutive,
        "intervention_first_epoch": int(interv_first_epoch),
        "freeze_epoch": int(freeze_epoch),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
