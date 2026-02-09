"""Plot overall accuracy curves from stagewise-accuracy pickles.

This script replicates the plotting behavior in src/composition_plots.ipynb, but
loads the time-series from the saved stagewise-accuracy pickle files (produced
by src/models/analyze_predictions.py with --save_stagewise_accuracies_only).

We specifically plot the metric:
  - predicted_exact_out_of_all_108

For each hop, we plot:
  - individual init_seed curves (lighter)
  - mean curve across init_seeds (darker/thicker)

Pickle structure expected:
  data = {
	"epochs": List[int] or np.ndarray,
	"seed_results": {
	  0: {"predicted_exact_out_of_all_108": List[float], ...}
	}
  }
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


METRIC_KEY = "predicted_exact_out_of_all_108"


# -------------------------
# Styling (match notebook)
# -------------------------
FIGSIZE: Tuple[int, int] = (12, 8)
LIGHT_ALPHA: float = 1.0 #.30
MEAN_ALPHA: float = 1.0
RUN_LINEWIDTH: float = 1.5
MEAN_LINEWIDTH: float = 3.0

LABEL_FONTSIZE: int = 26
TICK_FONTSIZE: int = 24
LEGEND_FONTSIZE: int = 14
LEGEND_LOC: str = "upper left"

SAVE_DPI: int = 300
SAVE_BBOX_INCHES: str = "tight"


HOP_COLORS: Dict[int, str] = {
	2: "blue",
	3: "green",
	4: "black",
	5: "red",
}


# -------------------------
# Import baseline pickle paths
# -------------------------
def _import_baseline_pickle_filepaths():
	"""Import baseline pickle path dicts from src/models/baseline_and_frozen_filepaths.py.

	src/models is not a Python package, so we add it to sys.path.
	"""

	this_dir = os.path.dirname(os.path.abspath(__file__))
	models_dir = os.path.join(this_dir, "models")
	if models_dir not in sys.path:
		sys.path.insert(0, models_dir)

	from baseline_and_frozen_filepaths import (  # type: ignore
		composition_baseline_pickle_file_paths,
		decomposition_baseline_pickle_file_paths,
	)

	return composition_baseline_pickle_file_paths, decomposition_baseline_pickle_file_paths


@dataclass(frozen=True)
class RunSeries:
	hop: int
	init_seed: int
	epochs: List[int]
	values: List[float]


def _load_pickle(path: str) -> dict:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Pickle not found: {path}")
	with open(path, "rb") as f:
		return pickle.load(f)


def _extract_series(data: dict, data_split_seed: int) -> Tuple[List[int], List[float]]:
	if "epochs" not in data:
		raise KeyError("Pickle missing key 'epochs'")
	if "seed_results" not in data:
		raise KeyError("Pickle missing key 'seed_results'")

	epochs_raw = data["epochs"]
	if isinstance(epochs_raw, np.ndarray):
		epochs = epochs_raw.tolist()
	else:
		epochs = list(epochs_raw)

	seed_results = data["seed_results"]
	if data_split_seed in seed_results:
		payload = seed_results[data_split_seed]
	elif str(data_split_seed) in seed_results:
		payload = seed_results[str(data_split_seed)]
	else:
		raise KeyError(
			f"seed_results missing data_split_seed={data_split_seed}; keys={list(seed_results.keys())}"
		)

	if METRIC_KEY not in payload:
		raise KeyError(f"Metric '{METRIC_KEY}' missing; available keys={list(payload.keys())}")

	values_raw = payload[METRIC_KEY]
	if not isinstance(values_raw, list):
		raise TypeError(f"Metric '{METRIC_KEY}' must be a list; got {type(values_raw)}")

	# guard for mismatch
	n = min(len(epochs), len(values_raw))
	return epochs[:n], values_raw[:n]


def _truncate_by_points(epochs: List[int], values: List[float], max_points: Optional[int]) -> Tuple[List[int], List[float]]:
	if max_points is None or max_points <= 0:
		return epochs, values
	return epochs[:max_points], values[:max_points]


def _mean_epoch_aligned(runs: Sequence[RunSeries], max_points: Optional[int]) -> Tuple[List[int], List[float]]:
	"""Compute nanmean across runs, aligned by epoch value (not index)."""
	all_epochs = sorted({e for r in runs for e in r.epochs})
	if max_points is not None and max_points > 0:
		all_epochs = all_epochs[:max_points]
	if not all_epochs:
		return [], []

	epoch_to_vals: Dict[int, List[float]] = {e: [] for e in all_epochs}
	for r in runs:
		ep_to_v = dict(zip(r.epochs, r.values))
		for e in all_epochs:
			epoch_to_vals[e].append(ep_to_v.get(e, np.nan))

	mean_vals = [float(np.nanmean(np.asarray(epoch_to_vals[e], dtype=float))) for e in all_epochs]
	return all_epochs, mean_vals


def _get_pickle_path_dict(exp_typ: str):
	composition_dict, decomposition_dict = _import_baseline_pickle_filepaths()
	if exp_typ == "composition":
		return composition_dict
	if exp_typ == "decomposition":
		return decomposition_dict
	raise ValueError(f"Unknown exp_typ: {exp_typ}")


def _available_init_seeds_for_hop(pickle_paths: dict, hop: int, data_split_seed: int) -> List[int]:
	try:
		hop_dict = pickle_paths[hop]
		seed_dict = hop_dict[data_split_seed]
	except Exception:
		return []

	seeds: List[int] = []
	for init_seed, p in seed_dict.items():
		if isinstance(p, str) and p.strip():
			seeds.append(int(init_seed))
	return sorted(seeds)


def plot_overall_accuracy(
	exp_typ: str,
	hops: Sequence[int],
	data_split_seed: int,
	init_seeds: Optional[Sequence[int]],
	max_points: Optional[int],
	output: Optional[str],
	strict: bool,
    seed_hp_values: Dict[int, str] = {},
	custom_colors_by_order: Optional[List[str]] = None,
):
	pickle_paths = _get_pickle_path_dict(exp_typ)

	fig, ax = plt.subplots(figsize=FIGSIZE)

	# Per-hop line styles for individual runs (as in notebook)
	per_run_linestyles = ["--", "-.", ":"]

	for hop in sorted(hops):
		if hop not in HOP_COLORS:
			print(f"[WARN] No color configured for hop={hop}; skipping")
			continue

		hop_color = HOP_COLORS[hop]

		available = _available_init_seeds_for_hop(pickle_paths, hop, data_split_seed)
		seeds_for_hop = list(init_seeds) if init_seeds is not None else available
		seeds_for_hop = [s for s in seeds_for_hop if s in available]

		if not seeds_for_hop:
			msg = f"No init_seeds available for hop={hop}, data_split_seed={data_split_seed}."
			if strict:
				raise ValueError(msg)
			print(f"[WARN] {msg} Skipping hop.")
			continue

		runs: List[RunSeries] = []
		for idx, seed in enumerate(seeds_for_hop):
			pkl_path = pickle_paths[hop][data_split_seed].get(seed, "")
			if not pkl_path:
				if strict:
					raise FileNotFoundError(f"Missing pickle path for hop={hop}, seed={seed}")
				print(f"[WARN] Missing pickle path for hop={hop}, seed={seed}; skipping")
				continue

			try:
				data = _load_pickle(pkl_path)
				epochs, values = _extract_series(data, data_split_seed=data_split_seed)
				epochs, values = _truncate_by_points(epochs, values, max_points)
				runs.append(RunSeries(hop=hop, init_seed=seed, epochs=epochs, values=values))
			except Exception as e:
				if strict:
					raise
				print(f"[WARN] Failed to load/parse hop={hop}, seed={seed}: {e}")
				continue

			# individual run line (light)
			wd, eta_min = seed_hp_values.get(seed, (None, None)) if seed in seed_hp_values else (None, None)
			if custom_colors_by_order is not None:
				hop_color = custom_colors_by_order[idx]
			ax.plot(
				epochs,
				values,
				color=hop_color,
				alpha=LIGHT_ALPHA,
				linewidth=RUN_LINEWIDTH if idx > 0 else RUN_LINEWIDTH + 1.0,
				linestyle=per_run_linestyles[idx % len(per_run_linestyles)] if idx > 0 else "-",
                label=f"wd {wd} eta_min {eta_min}"
			)

		if not runs:
			continue

		# mean curve (dark)
		# mean_epochs, mean_vals = _mean_epoch_aligned(runs, max_points=max_points)
		# ax.plot(
		# 	mean_epochs,
		# 	mean_vals,
		# 	color=hop_color,
		# 	alpha=MEAN_ALPHA,
		# 	linewidth=MEAN_LINEWIDTH,
		# 	label=f"Query hop = {hop}" if exp_typ == "composition" else f"Support hop = {hop}",
		# )

	# Styling (match notebook)
	ax.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
	ax.set_ylabel("Accuracy", fontsize=LABEL_FONTSIZE)
	ax.grid(True)
	ax.set_ylim(0, 1)
	ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
	ax.legend(fontsize=LEGEND_FONTSIZE, loc=LEGEND_LOC)

	if output:
		os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
		fig.savefig(output, bbox_inches=SAVE_BBOX_INCHES, dpi=SAVE_DPI)
		root, ext = os.path.splitext(output)
		if ext.lower() != ".pdf":
			fig.savefig(root + ".pdf", bbox_inches=SAVE_BBOX_INCHES, dpi=SAVE_DPI)
		print(f"[OK] Saved: {output}")
	else:
		plt.show()


def main():
	parser = argparse.ArgumentParser(
		description=(
			"Plot overall accuracy curves (predicted_exact_out_of_all_108) from stagewise pickles, "
			"with individual runs (light) and mean (dark) per hop."
		)
	)
	parser.add_argument(
		"--exp_typ",
		type=str,
		default="decomposition",
		choices=["composition", "decomposition"],
		help="Experiment type (selects which baseline pickle path dict to use).",
	)
	parser.add_argument(
		"--hops",
		type=int,
		nargs="+",
		default=[2, 3, 4, 5],
		help="Hop values to plot.",
	)
	parser.add_argument(
		"--data_split_seed",
		type=int,
		default=0,
		help="Data split seed key inside pickles (typically 0).",
	)
	parser.add_argument(
		"--init_seeds",
		type=int,
		nargs="+",
		default=None,
		help=(
			"Optional explicit init_seeds to include. If omitted, uses all available init_seeds per hop "
			"from baseline_and_frozen_filepaths.py."
		),
	)
	parser.add_argument(
		"--max_points",
		type=int,
		default=500,
		help="Plot only the first N points. Use <=0 to disable.",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="Output image path (e.g., out.png). Also writes a .pdf alongside.",
	)
	parser.add_argument(
		"--strict",
		action="store_true",
		help="If set, missing files/keys raise instead of being skipped.",
	)

	args = parser.parse_args()
	max_points = args.max_points if args.max_points and args.max_points > 0 else None
	
	weight_decay_hp_values = {
        3: ["0.1", "9e-5"],
        31: ["0.001", "8e-5"], 
        32: ["0.01", "8e-5"],
		33: ["0.1", "8e-5"],
    }
	custom_colors_by_order = ['darkgreen', 'green', 'green', 'green']

	plot_overall_accuracy(
		exp_typ=args.exp_typ,
		hops=args.hops,
		data_split_seed=args.data_split_seed,
		init_seeds=args.init_seeds,
		max_points=max_points,
		output=args.output,
		strict=args.strict,
		seed_hp_values=weight_decay_hp_values,
		custom_colors_by_order=custom_colors_by_order,
	)


if __name__ == "__main__":
	main()