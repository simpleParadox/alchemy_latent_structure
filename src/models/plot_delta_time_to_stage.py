import argparse
import json
import re
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

METRIC_KEYS = {
    "p_b_given_a": "predicted_in_context_correct_half_accuracies",
    "p_a": "predicted_in_context_accuracies",
    "p_c_given_ab": "predicted_in_context_correct_half_exact_accuracies",
}


def natural_sort_key(s: str):
    """Sort strings containing numbers naturally (e.g., 'Layer 2' comes before 'Layer 10')."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", s)]


def _cap_delta_t(delta_t: Optional[float], cap: float) -> Optional[float]:
    if delta_t is None:
        return None
    try:
        v = float(delta_t)
    except (TypeError, ValueError):
        return None
    return min(v, cap)


def _bin_center(x: float, bin_width: int) -> int:
    if bin_width <= 0:
        raise ValueError("bin_width must be >= 1")
    return int(np.round(x / bin_width) * bin_width)


def _get_t_baseline_from_payload(payload: dict) -> Optional[int]:
    """
    Prefer directly stored t_baseline in the payload.
    """
    tb = payload.get("t_baseline", None)
    if tb is None:
        return None
    try:
        return int(tb)
    except (TypeError, ValueError):
        return None


def extract_plot_data_with_errorbars(
    results_json_path: str,
    target_metric_key: str,
    exp_typ: str = "held_out",
    data_split_seed: int = 0,
    init_seeds: Optional[List[int]] = None,
    cap_never_reached: float = 1000.0,
    min_n: int = 2,
    x_mode: str = "absolute",              # NEW
    bin_width: int = 10,                  # NEW (used for relative)
    anchor_metric_key: Optional[str] = None,  # NEW (defaults to target_metric_key)
) -> Dict[str, List[Tuple[int, float, float, int]]]:
    """
    Returns:
      layer_data[layer] = list of (x, mean_delta_t, sem_delta_t, n)
    where x is:
      - freeze_epoch (absolute)  OR
      - binned (freeze_epoch - t_baseline[anchor_metric]) (relative)
    """
    if init_seeds is None:
        init_seeds = [1, 3, 42]

    if anchor_metric_key is None:
        anchor_metric_key = target_metric_key

    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ds_key = str(data_split_seed)
    if ds_key not in data:
        print(f"Warning: data_split_seed '{ds_key}' not found in {results_json_path}")
        return {}

    # Collect values: layer -> epoch -> list[delta_t across seeds]
    values: Dict[str, Dict[int, List[float]]] = {}

    for init_seed in init_seeds:
        init_key = str(init_seed)
        if init_key not in data[ds_key]:
            print(f"Warning: init_seed '{init_key}' not found under data_split_seed '{ds_key}'. Skipping.")
            continue

        run_json = data[ds_key][init_key]
        results = run_json.get("results", {})
        if exp_typ not in results:
            print(f"Warning: exp_typ '{exp_typ}' missing for data_split_seed={ds_key}, init_seed={init_key}. Skipping.")
            continue

        exp_results = results[exp_typ]  # layer -> epoch -> metric -> payload
        for layer_name, epochs_data in exp_results.items():
            for epoch_str, metrics in epochs_data.items():
                if target_metric_key not in metrics:
                    continue
                payload = metrics[target_metric_key]

                # Prefer freeze_epoch from payload; fall back to epoch_str
                freeze_epoch = payload.get("freeze_epoch")
                if freeze_epoch is None:
                    try:
                        freeze_epoch = int(epoch_str)
                    except ValueError:
                        raise ValueError(f"Invalid epoch string: {epoch_str}")

                delta_t = _cap_delta_t(payload.get("delta_t"), cap_never_reached)
                if delta_t is None:
                    continue

                if anchor_metric_key is None:
                    anchor_metric_key = target_metric_key

                if x_mode == "absolute":
                    x_val = int(freeze_epoch)

                elif x_mode == "relative":
                    # print(f"Computing relative x for layer={layer_name}, epoch={freeze_epoch}, init_seed={init_seed}")
                    # We need t_baseline for the anchor metric.
                    # 1) If anchor_metric_key == target_metric_key, it's in the same payload.
                    # 2) If different, try to read it from the corresponding anchor payload at same (layer, epoch),
                    #    else skip (or you could implement a more thorough lookup).
                    if anchor_metric_key == target_metric_key:
                        t_base = _get_t_baseline_from_payload(payload)
                    else:
                        anchor_payload = metrics.get(anchor_metric_key, {})
                        t_base = _get_t_baseline_from_payload(anchor_payload)

                    if t_base is None:
                        raise ValueError(
                            f"t_baseline for anchor_metric_key='{anchor_metric_key}' not found "
                            f"at layer='{layer_name}', epoch={freeze_epoch}, init_seed={init_seed}."
                        )

                    rel = float(freeze_epoch) - float(t_base)
                    # print(f"  t_baseline={t_base}, relative freeze epoch={rel}")
                    x_val = _bin_center(rel, bin_width=bin_width)

                else:
                    raise ValueError(f"Unknown x_mode={x_mode}. Use 'absolute' or 'relative'.")

                values.setdefault(layer_name, {}).setdefault(int(x_val), []).append(float(delta_t))

    # Aggregate to mean + SEM
    layer_data: Dict[str, List[Tuple[int, float, float, int]]] = {}
    for layer_name, epoch_map in values.items():
        series = []
        # import pdb; pdb.set_trace()
        for x_key, vals in epoch_map.items():
            n = len(vals)
            if n < min_n:
                continue
            arr = np.asarray(vals, dtype=float)
            mean = float(np.mean(arr))
            # SEM: std / sqrt(n); use ddof=1 only if n>1
            std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            sem = float(std / np.sqrt(n)) if n > 1 else 0.0

            x = int(x_key)
            series.append((x, mean, sem, n))

        if series:
            series.sort(key=lambda x: x[0])
            layer_data[layer_name] = series

    return layer_data


def plot_deltas_with_errorbars(
    results_path: str,
    target_metric: str,
    epsilon: float = 10,
    exp_typ: str = "held_out",
    output_path: Optional[str] = None,
    stage_key: str = "p_b_given_a",
    data_split_seed: int = 0,
    init_seeds: Optional[List[int]] = None,
    min_n: int = 2,
    cap_never_reached: float = 1000.0,
    x_mode: str = "absolute",       # NEW
    bin_width: int = 10,            # NEW
    anchor_metric: Optional[str] = None,  # NEW (resolved metric string)
    anchor_key: Optional[str] = None,     # NEW (alias)
):
    layer_data = extract_plot_data_with_errorbars(
        results_json_path=results_path,
        target_metric_key=target_metric,
        exp_typ=exp_typ,
        data_split_seed=data_split_seed,
        init_seeds=init_seeds,
        cap_never_reached=cap_never_reached,
        min_n=min_n,
        x_mode=x_mode,
        bin_width=bin_width,
        anchor_metric_key=anchor_metric,
    )

    if not layer_data:
        print(
            f"No data available to plot for metric '{target_metric}' exp_typ='{exp_typ}', "
            f"data_split_seed={data_split_seed} (min_n={min_n})."
        )
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    sorted_layers = sorted(layer_data.keys(), key=natural_sort_key)
    for layer in sorted_layers:
        series = layer_data[layer]
        epochs = [p[0] for p in series]
        means = [p[1] for p in series]
        sems = [p[2] for p in series]
        ns = [p[3] for p in series]
        # import pdb; pdb.set_trace()

        ax.errorbar(
            epochs,
            means,
            yerr=sems,
            marker="o",
            linestyle="-",
            capsize=3,
            label=f"{layer}",
        )

        # Optional: annotate n at each point (kept off by default to avoid clutter)
        # for x, y, n in zip(epochs, means, ns):
        #     ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax.axhline(y=epsilon, color="black", linestyle="--", label=f"Critical epoch threshold ({epsilon})")

    if x_mode == "absolute":
        ax.set_xlabel("Freeze Epoch", fontsize=18)
    else:
        ax.set_xlabel(
            f"Relative Freeze Epoch (t_f - t_base[{anchor_key or stage_key}]) [bin={bin_width}]",
            fontsize=18,
        )
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.8)

    init_seed_str = ",".join(str(s) for s in (init_seeds if init_seeds is not None else [1, 3, 42]))
    ax.set_ylabel("Δt (Delay to Event 'k')", fontsize=18)
    ax.set_title(
        "Impact of Freezing on the Delay of Event 'k'\n"
        # f"exp_typ={exp_typ} | metric={stage_key} | data_split_seed={data_split_seed} | init_seeds=[{init_seed_str}]"
        f"exp_typ={exp_typ} | event={stage_key}",
        fontsize=20,
    )
    # ax.legend(title="Frozen Layer", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.legend(title="Frozen Layer", loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    # x-ticks font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Delta T from JSON results (with SEM error bars across init seeds).")

    parser.add_argument(
        "--json_path",
        type=str,
        default="delta_time_results.json",
        help="Path to the results JSON file.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="p_c_given_ab",
        help="Metric key alias to plot (one of: p_b_given_a, p_a, p_c_given_ab).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=10,
        help="Epsilon threshold for horizontal line.",
    )
    parser.add_argument(
        "--exp_typ",
        type=str,
        default="held_out",
        choices=["held_out", "composition", "decomposition"],
        help="Experiment type to plot.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the plot image (optional).",
    )

    # NEW: JSON scoping + aggregation controls
    parser.add_argument(
        "--data_split_seed",
        type=int,
        default=0,
        help="Which data_split_seed to plot from the JSON (top-level key).",
    )
    parser.add_argument(
        "--init_seeds",
        type=int,
        nargs="+",
        default=[1, 3, 42],
        help="Init seeds to aggregate over (second-level keys under data_split_seed).",
    )
    parser.add_argument(
        "--min_n",
        type=int,
        default=1,
        help="Only plot a point if at least min_n init seeds have data for that (layer, epoch).",
    )
    parser.add_argument(
        "--cap_never_reached",
        type=float,
        default=1000.0,
        help="Cap delta_t at this value (used for 'never reached' sentinel).",
    )
    parser.add_argument(
        "--x_mode",
        type=str,
        default="relative",
        choices=["absolute", "relative"],
        help="Use absolute freeze epochs or relative (freeze_epoch - t_baseline) with optional binning.",
    )
    parser.add_argument(
        "--bin_width",
        type=int,
        default=10,
        help="Bin width for relative x_mode (ignored for absolute).",
    )
    parser.add_argument(
        "--anchor_metric",
        type=str,
        default='p_a',
        help="Metric alias used as the baseline anchor (t_baseline). Defaults to --metric if not provided.",
    )

    args = parser.parse_args()

    metric = METRIC_KEYS.get(args.metric, None)
    assert metric is not None, f"Unknown metric key alias: {args.metric}. Known: {sorted(METRIC_KEYS.keys())}"

    anchor_alias = args.anchor_metric if args.anchor_metric is not None else args.metric
    anchor_metric = METRIC_KEYS.get(anchor_alias, None)
    assert anchor_metric is not None, f"Unknown anchor_metric alias: {anchor_alias}. Known: {sorted(METRIC_KEYS.keys())}"

    plot_deltas_with_errorbars(
        results_path=args.json_path,
        target_metric=metric,
        epsilon=args.epsilon,
        exp_typ=args.exp_typ,
        output_path=args.output,
        stage_key=args.metric,
        data_split_seed=args.data_split_seed,
        init_seeds=args.init_seeds,
        min_n=args.min_n,
        cap_never_reached=args.cap_never_reached,
        x_mode=args.x_mode,
        bin_width=args.bin_width,
        anchor_metric=anchor_metric,
        anchor_key=anchor_alias,
    )


if __name__ == "__main__":
    main()