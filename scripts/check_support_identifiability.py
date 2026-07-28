#!/usr/bin/env python3
"""
Identifiability gate for the length-/support-matched decomposition control.

For each episode, checks how many of the full candidate chemistries are consistent
with the observed support set, and what fraction of queries have a unique predicted
answer across those consistent chemistries. Must be run on the stratified 48-example
data before any training launches (see prompts/implement.md gate), and on the
full-enumeration data as a sanity control (must return ~100% determined there, or the
checker itself is broken).

Reads the generated support/query JSON produced by
src/data/support_and_query_generator.py, plus the source chemistry pickle/JSON it was
generated from (for the full candidate-chemistry pool).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "data"))
from support_and_query_generator import load_chemistry_graph  # noqa: E402


def build_transition_lookups(chemistries: Dict) -> Dict[str, Dict[str, Dict[str, str]]]:
    """transition[chemistry_id][node_id][potion_color] -> next_node_id, for every
    complete chemistry in the pool."""
    lookups = {}
    for ep_id, ep_data in chemistries.items():
        if ep_id == "_metadata" or not ep_data.get("is_complete", False):
            continue
        graph = ep_data["graph"]
        node_lookup = {}
        for node_id, node_data in graph.items():
            node_lookup[node_id] = {
                t["potion_color"]: t["next_node_str"] for t in node_data.get("transitions", [])
            }
        lookups[ep_id] = node_lookup
    return lookups


def simulate(lookup: Dict[str, Dict[str, str]], start_node: str, potions: List[str]) -> Optional[str]:
    """Apply a potion sequence from start_node under one chemistry's transition lookup.
    Returns the resulting node id, or None if the sequence isn't a valid path under
    this chemistry."""
    cur = start_node
    for p in potions:
        nxt = lookup.get(cur, {}).get(p)
        if nxt is None:
            return None
        cur = nxt
    return cur


def consistent_chemistries(
    lookups: Dict[str, Dict[str, Dict[str, str]]],
    support_infos: List[Dict],
) -> List[str]:
    consistent = []
    for c_id, lookup in lookups.items():
        ok = True
        for s_info in support_infos:
            end = simulate(lookup, s_info["start_node"], s_info["potions"])
            if end != s_info["end_node"]:
                ok = False
                break
        if ok:
            consistent.append(c_id)
    return consistent


def run_check(chemistries_path: str, data_path: str, label: str, num_episodes: Optional[int]) -> Dict:
    print(f"[{label}] Loading chemistries from {chemistries_path}...")
    chemistries = load_chemistry_graph(chemistries_path)
    lookups = build_transition_lookups(chemistries)
    print(f"[{label}] Built transition lookups for {len(lookups)} complete chemistries.")

    with open(data_path, "r") as f:
        data = json.load(f)
    episodes = data["episodes"]
    episode_ids = list(episodes.keys())
    if num_episodes is not None:
        episode_ids = episode_ids[:num_episodes]

    consistent_counts = []
    total_queries = 0
    determined_queries = 0

    for ep_id in episode_ids:
        if ep_id not in lookups:
            print(f"[{label}] WARNING: episode {ep_id} in data but not in chemistry pool, skipping.")
            continue
        ep = episodes[ep_id]
        support_infos = ep["support_samples_info"]
        query_infos = ep["query_samples_info"]

        consistent = consistent_chemistries(lookups, support_infos)
        consistent_counts.append(len(consistent))

        for q_info in query_infos:
            x = q_info["start_node"]
            preds = set()
            for c_id in consistent:
                pred = simulate(lookups[c_id], x, q_info["potions"])
                if pred is not None:
                    preds.add(pred)
            total_queries += 1
            if len(preds) == 1:
                determined_queries += 1

    pct_determined = 100.0 * determined_queries / total_queries if total_queries else float("nan")
    result = {
        "label": label,
        "num_episodes_checked": len(consistent_counts),
        "consistent_count_mean": statistics.mean(consistent_counts) if consistent_counts else float("nan"),
        "consistent_count_median": statistics.median(consistent_counts) if consistent_counts else float("nan"),
        "consistent_count_max": max(consistent_counts) if consistent_counts else float("nan"),
        "total_queries": total_queries,
        "determined_queries": determined_queries,
        "pct_determined": pct_determined,
    }

    print(f"\n=== Identifiability report: {label} ===")
    print(f"Episodes checked: {result['num_episodes_checked']}")
    print(
        "|Consistent(S)|: mean={:.2f}, median={:.1f}, max={}".format(
            result["consistent_count_mean"], result["consistent_count_median"], result["consistent_count_max"]
        )
    )
    print(f"Queries determined: {determined_queries}/{total_queries} ({pct_determined:.2f}%)")
    return result


def main():
    parser = argparse.ArgumentParser(description="Identifiability gate for support-matched decomposition data.")
    parser.add_argument("--chemistries", required=True, help="Path to the full chemistry pool (pkl/json/gz).")
    parser.add_argument("--data", required=True, help="Path to the generated train/val JSON to check.")
    parser.add_argument("--label", default=None, help="Label for the report (defaults to --data basename).")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Cap the number of episodes checked (default: all in the file).")
    args = parser.parse_args()

    label = args.label or Path(args.data).name
    run_check(args.chemistries, args.data, label, args.num_episodes)


if __name__ == "__main__":
    main()
