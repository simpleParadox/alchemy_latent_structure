"""
STEP 0 prerequisite test for the half-edge potion-pairing CL experiment
(see CL_experiment_handoff.md sec 2).

Checks whether the existing D1 (pairing_index=0) generated dataset is invariant
under the renaming ORANGE <-> YELLOW. This must hold for SIGMA = (GREEN YELLOW)
to be the exact minimal repair mapping pairing_0 -> pairing_6. If it fails, the
chemistry sampler has a direction-assignment asymmetry between the two halves
of the (RED,ORANGE)/(YELLOW,?) axis and the whole design is invalid.

IMPORTANT: the full chemistry set (all 48 potion_maps) is (OY)-closed by
construction, but a random train/val split is not -- a chemistry c landing in
train and its (OY)-counterpart c' landing in val makes both splits look
individually non-invariant even though the underlying generator is fine. All
provided files are therefore merged into one multiset BEFORE comparison; do
not check splits individually.

Usage:
    python src/analysis/test_oy_invariance.py <path_to_train_json> <path_to_val_json> ...
"""
import sys
import json
import re
from collections import Counter

RENAME = {"ORANGE": "YELLOW", "YELLOW": "ORANGE"}
POTION_RE = re.compile(r"\b(RED|GREEN|ORANGE|YELLOW|PINK|CYAN)\b")


def rename_transition(sample_str: str) -> str:
    return POTION_RE.sub(lambda m: RENAME.get(m.group(0), m.group(0)), sample_str)


def collect_multiset(path: str) -> Counter:
    with open(path, "r") as f:
        data = json.load(f)
    counter = Counter()
    for episode in data["episodes"].values():
        for s in episode["support"]:
            counter[s] += 1
        for s in episode["query"]:
            counter[s] += 1
    return counter


def renamed_multiset(counter: Counter) -> Counter:
    renamed = Counter()
    for s, count in counter.items():
        renamed[rename_transition(s)] += count
    return renamed


def check_invariance(paths) -> bool:
    combined = Counter()
    for p in paths:
        combined.update(collect_multiset(p))

    renamed = renamed_multiset(combined)
    total = sum(combined.values())
    print(f"Merged {len(paths)} file(s), {total} total transitions.")

    if combined == renamed:
        print("PASS: merged dataset is invariant under ORANGE<->YELLOW renaming.")
        return True

    print("FAIL: merged dataset is NOT invariant under ORANGE<->YELLOW renaming.")
    only_in_original = combined - renamed
    only_in_renamed = renamed - combined
    print(f"  Transitions in original but not reproduced after renaming ({len(only_in_original)} distinct):")
    for s, count in list(only_in_original.items())[:10]:
        print(f"    x{count}  {s}")
    print(f"  Transitions produced by renaming but absent from original ({len(only_in_renamed)} distinct):")
    for s, count in list(only_in_renamed.items())[:10]:
        print(f"    x{count}  {s}")
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/analysis/test_oy_invariance.py <json_file> [<json_file> ...]")
        print("Pass ALL splits (train + val) of the same pairing_index together --")
        print("checking a single split in isolation gives false failures (see module docstring).")
        sys.exit(1)

    paths = sys.argv[1:]
    passed = check_invariance(paths)

    if passed:
        print("\nOVERALL: PASS. SIGMA=(GREEN YELLOW) is the exact minimal repair. Proceed with Step 1.")
        sys.exit(0)
    else:
        print("\nOVERALL: FAIL. STOP. Report the asymmetry before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
