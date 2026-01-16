"""
delta_time_to_stage.py

Minimal delta-t (delay) computation for event curves saved as stagewise .pkl.

Each .pkl must contain:
  - epochs: 1D array (E,)
  - one or more metric arrays, each 1D (E,)

This script computes:
  delta_t = t_intervention - t_baseline

where t is the first epoch where the chosen metric >= tau for `consecutive`
consecutive evaluation points.

The output JSON is structured as:
  results -> exp_typ -> layer -> epoch -> metric_key -> { values }

Usage
-----
# Compute delta-t for chosen metric keys (single or multiple)
python src/models/delta_time_to_stage.py \
  --baseline_pickle /abs/path/to/baseline_metrics.pkl \
  --intervention_pickle /abs/path/to/freeze_layer0_metrics.pkl \
  --metric_key EVENT_B_GIVEN_A EVENT_C_GIVEN_AB \
  --tau 0.90 --consecutive 3 \
  --output_json /abs/path/to/delta_t_results.json

# List available keys in a .pkl
python src/models/delta_time_to_stage.py \
  --baseline_pickle /abs/path/to/baseline_metrics.pkl \
  --list_keys
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Any
import pickle
import numpy as np


from baseline_and_frozen_filepaths import (
    staged_accuracies_held_out_file_paths_baseline_pickles,
    staged_accuracies_held_out_file_paths_frozen_pickles,
)


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


def list_metric_keys(payload: dict, seed: int) -> List[str]:
    if "seed_results" not in payload:
        raise KeyError("payload missing required key 'seed_results'")
    if seed not in payload["seed_results"]:
        raise KeyError(f"Seed {seed} not found. Available: {sorted(payload['seed_results'].keys())}")
    return sorted(payload["seed_results"][seed].keys())


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
        help="Absolute path to baseline .pkl (must contain 'epochs' and metric arrays).",
    )
    p.add_argument(
        "--intervention_pickle",
        type=str,
        nargs="*",
        default=None,
        help="One or more intervention .pkl files. Required unless --list_keys is used.",
    )

    p.add_argument(
        "--metric_key",
        nargs="+",
        default=["EVENT_B_GIVEN_A", "EVENT_C_GIVEN_AB", "EVENT_A"],
        help=(
            "Metric key(s) to use for stage time. You may pass either keys in METRIC_KEYS "
            "(e.g., EVENT_C_GIVEN_AB) or raw .pkl keys (e.g., p_b_given_a). "
            "You can pass multiple values separated by spaces."
        ),
    )

    p.add_argument("--tau", type=float, default=0.90)
    p.add_argument("--consecutive", type=int, default=3)

    p.add_argument(
        "--list_keys",
        action="store_true",
        help="If set, list keys in --baseline_pickle and exit.",
    )

    p.add_argument('--seed', type=int, default=42, help='Seed index to use from the .pkl files.')
    p.add_argument('--data_split_seed', type=int, default=0, help='Data split seed used in the experiments.')
    p.add_argument('--init_seed', type=int, default=42, help='Model initialization seed used in the experiments.')
    p.add_argument(
        "--output_json",
        type=str,
        default="delta_time_results.json",
        help="Path to save or update the delta-t results JSON.",
    )
    p.add_argument(
        "--calc_baseline_only",
        action="store_true",
        help="If set, only compute t_baseline for the given metrics and exit (skips intervention pickles).",
    )
    p.add_argument(
        "--exp_typ",
        type=str,
        default="held_out",
        choices=["held_out", "composition", "decomposition"],
        help="Type of experiment for frozen pickles.",
    )

    return p.parse_args()


def resolve_metric_keys(raw_keys: Iterable[str]) -> List[str]:
    resolved = []
    for key in raw_keys:
        if "," in key:
            for part in key.split(","):
                part = part.strip()
                if part:
                    resolved.append(METRIC_KEYS.get(part, part))
        else:
            resolved.append(METRIC_KEYS.get(key, key))
    return resolved


def collect_frozen_pickles(data_split_seed: int, init_seed: int,
    base_path: str, frozen_epochs: List[int], frozen_layers: List[str],
    exp_typ: str) -> List[Dict[str, Any]]:
    
    results: List[Dict[str, Any]] = []

    """
    Example: /home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/stagewise_accuracies_frozen_layer_transformer_layer_0_freeze_epoch_100_data_split_seed_0_init_seed_42_hop_4_exp_held_out.pkl
    """
    
    # Generate list of (path, layer, epoch) tuples first
    candidates = []
    
    if exp_typ == 'held_out':
        for layer in frozen_layers:
            for epoch in frozen_epochs:
                path = f"{base_path}/stagewise_accuracies_frozen_layer_{layer}_freeze_epoch_{epoch}_data_split_seed_{data_split_seed}_init_seed_{init_seed}_hop_4_exp_held_out_relative_epoch.pkl"
                candidates.append((path, layer, epoch))
                
    elif exp_typ == 'composition':
        raise NotImplementedError("Composition frozen pickle collection not implemented yet.")

    elif exp_typ == 'decomposition':
        raise NotImplementedError("Decomposition frozen pickle collection not implemented yet.")

    for (path, layer, epoch) in candidates:
        if os.path.exists(path):
            results.append({
                "path": path,
                "layer": layer,
                "epoch": epoch,
                "exp_typ": exp_typ
            })
        else:
            print(f"Warning: Frozen pickle not found at {path}, skipping.")
            
    return results   


def parse_filename_metadata(path: str, default_exp_typ: str) -> Dict[str, Any]:
    """
    Attempts to extract layer, epoch, and experiment type from a manually provided filename.
    Useful when files are provided via CLI args instead of auto-collected.
    """
    filename = os.path.basename(path)
    # Regex designed to capture 'frozen_layer_X' and 'freeze_epoch_Y'
    # Flexible enough to catch simple variations
    layer_match = re.search(r"frozen_layer_(.+?)_freeze_epoch", filename)
    epoch_match = re.search(r"freeze_epoch_(\d+)", filename)
    
    layer = layer_match.group(1) if layer_match else "unknown_layer"
    epoch = int(epoch_match.group(1)) if epoch_match else -1
    
    return {
        "path": path,
        "layer": layer,
        "epoch": epoch,
        "exp_typ": default_exp_typ  # Assume user provided type applies
    }


def compute_delta_t(
    base_payload: dict,
    int_payload: dict,
    metric_key: str,
    seed: int,
    spec: StageSpec,
    data_split_seed: Optional[int] = None,
) -> dict:
    if "epochs" not in base_payload:
        raise KeyError("baseline pickle missing required key 'epochs'")
    if "epochs" not in int_payload:
        raise KeyError("intervention pickle missing required key 'epochs'")

    if metric_key not in base_payload["seed_results"][data_split_seed]:
        raise KeyError(f"baseline pickle missing metric '{metric_key}'")
    if metric_key not in int_payload["seed_results"][data_split_seed]:
        raise KeyError(f"intervention pickle missing metric '{metric_key}'")

    base_epochs = np.asarray(base_payload["epochs"]).astype(int)[1:]
    int_epochs = np.asarray(int_payload["epochs"]).astype(int)[1:]
    base_values = np.asarray(base_payload["seed_results"][data_split_seed][metric_key]).astype(float)
    int_values = np.asarray(int_payload["seed_results"][data_split_seed][metric_key]).astype(float)

    t_base = first_sustained_crossing(base_epochs, base_values, spec)
    if t_base is None:
        raise RuntimeError(
            f"Baseline never reached stage for metric={metric_key} with tau={spec.tau}, consecutive={spec.consecutive}."
        )

    interv_first_epoch = int(np.min(int_epochs)) if int_epochs.size > 0 else None
    freeze_epoch = None
    if interv_first_epoch is not None:
        freeze_epoch = interv_first_epoch - 1

    if int_epochs.size > 0:
        assert int_epochs.min() == freeze_epoch + 1, (
            f"Intervention epochs should start at freeze_epoch + 1 = {freeze_epoch + 1}, "
            f"but got min epoch = {int_epochs.min()}"
        )

    if t_base is not None and freeze_epoch is not None and int(t_base) <= int(freeze_epoch):
        t_int = t_base
    else:
        prefix_needed = max(spec.consecutive - 1, 0)
        prefix_mask = base_epochs < (interv_first_epoch if interv_first_epoch is not None else 0)
        base_prefix_epochs = (
            base_epochs[prefix_mask][-prefix_needed:]
            if prefix_needed > 0 and prefix_mask.any()
            else np.array([], dtype=int)
        )

        if base_prefix_epochs.size > 0:
            base_map = {int(e): float(v) for e, v in zip(base_epochs.tolist(), base_values.tolist())}
            prefix_vals = np.array([base_map[int(e)] for e in base_prefix_epochs.tolist()], dtype=float)
        else:
            prefix_vals = np.array([], dtype=float)

        hybrid_epochs = np.concatenate([base_prefix_epochs, int_epochs]).astype(int)
        hybrid_values = np.concatenate([prefix_vals, int_values]).astype(float)
        t_int = first_sustained_crossing(hybrid_epochs, hybrid_values, spec)

    if t_int is None:
        return {
            "metric_key": metric_key,
            "t_baseline": int(t_base),
            "t_intervention": None,
            "delta_t": 100000,
            "tau": spec.tau,
            "consecutive": spec.consecutive,
            "note": "intervention never reached stage within available epochs",
            "intervention_first_epoch": int(interv_first_epoch) if interv_first_epoch is not None else None,
            "freeze_epoch": int(freeze_epoch) if freeze_epoch is not None else None,
            "init_seed": seed, 
            "data_split_seed": data_split_seed,
        }

    return {
        "metric_key": metric_key,
        "t_baseline": int(t_base),
        "t_intervention": int(t_int),
        "delta_t": int(t_int - t_base),
        "tau": spec.tau,
        "consecutive": spec.consecutive,
        "intervention_first_epoch": int(interv_first_epoch) if interv_first_epoch is not None else None,
        "freeze_epoch": int(freeze_epoch) if freeze_epoch is not None else None,
        "init_seed": seed,
        "data_split_seed": data_split_seed,
    }


def load_results_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()

    if args.exp_typ == 'held_out':
        baseline_file_paths = staged_accuracies_held_out_file_paths_baseline_pickles
    elif args.exp_typ == 'composition':
        raise NotImplementedError("Composition baseline pickle path not implemented yet.")
    elif args.exp_typ == 'decomposition':
        raise NotImplementedError("Decomposition baseline pickle path not implemented yet.")

    if not args.baseline_pickle:
        args.baseline_pickle = baseline_file_paths.get(
            args.data_split_seed, {}
        ).get(args.init_seed)

    if not args.baseline_pickle:
        raise ValueError("--baseline_pickle is required unless configured in baseline_and_frozen_filepaths.py")

    base_payload = load_stagewise_pickle(args.baseline_pickle)

    if args.list_keys:
        print(
            json.dumps(
                {
                    "file": args.baseline_pickle,
                    "keys": list_metric_keys(base_payload, args.init_seed),
                    "placeholder_keys": METRIC_KEYS,
                },
                indent=2,
            )
        )
        return

    metric_keys = resolve_metric_keys(args.metric_key)
    spec = StageSpec(tau=float(args.tau), consecutive=int(args.consecutive))

    # --- NEW: scope results by (data_split_seed, init_seed) ---
    results_json = load_results_json(args.output_json)

    ds_key = str(args.data_split_seed)
    init_key = str(args.init_seed)

    ds_dict = results_json.setdefault(ds_key, {})
    run_json = ds_dict.setdefault(init_key, {})

    # Always write/run-specific metadata (do not use setdefault)
    run_json["baseline_pickle"] = args.baseline_pickle
    run_json["data_split_seed"] = args.data_split_seed
    run_json["init_seed"] = args.init_seed
    run_json["tau"] = spec.tau
    run_json["consecutive"] = spec.consecutive
    run_json["exp_typ"] = args.exp_typ

    print("Baseline pickle:", args.baseline_pickle)

    if args.calc_baseline_only:
        print("Calculating baseline stages only...")
        baseline_stages = run_json.setdefault("baseline_stages", {})

        # Verify data_split_seed exists in baseline payload
        if args.data_split_seed not in base_payload["seed_results"]:
            raise KeyError(f"Data split seed {args.data_split_seed} not found in baseline pickle.")

        base_epochs = np.asarray(base_payload["epochs"]).astype(int)[1:]

        for metric_key in metric_keys:
            if metric_key not in base_payload["seed_results"][args.data_split_seed]:
                print(
                    f"Warning: Metric '{metric_key}' not found in baseline for data_split_seed {args.data_split_seed}. Skipping."
                )
                continue

            base_values = np.asarray(base_payload["seed_results"][args.data_split_seed][metric_key]).astype(float)
            t_base = first_sustained_crossing(base_epochs, base_values, spec)
            baseline_stages[metric_key] = int(t_base) if t_base is not None else None

        # save_results_json(args.output_json, results_json)
        print(json.dumps(run_json, indent=2))
        return

    intervention_items: List[Dict[str, Any]] = []

    if args.intervention_pickle:
        for p in args.intervention_pickle:
            intervention_items.append(parse_filename_metadata(p, args.exp_typ))
    else:
        # frozen_epochs = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200] # These were absolute epochs - do not use.
        # frozen_epochs = np.arange(100, 301, 10).tolist()

        # init seed 42 relative epochs.
        if args.init_seed == 42:
            frozen_epochs = [68, 78, 88, 98, 108, 118, 128, 138, 148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 248, 258, 268, 278, 288, 298, 308]

        # init seed 3 relative epochs.
        elif args.init_seed == 3:
            frozen_epochs = [72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 232, 242, 252, 262, 272, 282, 292, 302, 312]

        # init seed 1 relative epochs.
        elif args.init_seed == 1:
            frozen_epochs = [82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 232, 242, 252, 262, 272, 282, 292, 302, 312, 322]
        else:
            raise ValueError(f"Relative frozen epochs not defined for init_seed {args.init_seed}")
        frozen_layers = [f"transformer_layer_{i}" for i in range(4)]
        frozen_layers.insert(0, "embedding_layer")

        if args.exp_typ == 'held_out':
            base_path = staged_accuracies_held_out_file_paths_frozen_pickles.get(
                args.data_split_seed, {}
            ).get(args.init_seed)['base_path']
        elif args.exp_typ == 'composition':
            base_path = None
            raise NotImplementedError("Composition frozen pickle base path not implemented yet.")
        elif args.exp_typ == 'decomposition':
            base_path = None
            raise NotImplementedError("Decomposition frozen pickle base path not implemented yet.")

        assert base_path is not None, "Base path for frozen pickles not found in configuration."

        intervention_items = collect_frozen_pickles(
            args.data_split_seed, args.init_seed,
            base_path=base_path,
            frozen_epochs=frozen_epochs,
            frozen_layers=frozen_layers,
            exp_typ=args.exp_typ
        )

    if not intervention_items:
        raise ValueError("--intervention_pickle is required unless frozen pickles can be collected from config")

    # Ensure results container exists within this run
    run_json.setdefault("results", {})

    for item in intervention_items:
        path = item["path"]
        layer_name = item["layer"]
        epoch_val = item["epoch"]
        exp_typ = item["exp_typ"]

        try:
            int_payload = load_stagewise_pickle(path)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            print(f"Error loading pickle {path}: {e}. Skipping.")
            continue

        exp_dict = run_json["results"].setdefault(exp_typ, {})
        layer_dict = exp_dict.setdefault(layer_name, {})
        epoch_dict = layer_dict.setdefault(str(epoch_val), {})

        for metric_key in metric_keys:
            computed_result = compute_delta_t(
                base_payload=base_payload,
                int_payload=int_payload,
                metric_key=metric_key,
                seed=args.init_seed,
                data_split_seed=args.data_split_seed,
                spec=spec,
            )
            computed_result["intervention_pickle"] = path
            epoch_dict[metric_key] = computed_result

    save_results_json(args.output_json, results_json)
    print(f"Saved delta-t results to {args.output_json}")
    # print(json.dumps(run_json, indent=2))


if __name__ == "__main__":
    main()
