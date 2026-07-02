import json
import re

baseline_train = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/baseline_generated_data_from_normalized_reward/compositional_baseline_compositional_chemistry_samples_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0.json"

with open(baseline_train, 'r') as f:
    data = json.load(f)

eps = data['episodes']
ep_ids = list(eps.keys())
print(f"Total episodes: {len(ep_ids)}")
print(f"First 3 episode IDs: {ep_ids[:3]}")

first_ep = eps[ep_ids[0]]
support = first_ep.get('support', [])
query = first_ep.get('query', [])
print(f"First episode support count: {len(support)}")
print(f"First episode query count: {len(query)}")
print(f"First support example: {support[0][:150] if support else 'EMPTY'}")
print(f"First query example: {query[0][:150] if query else 'EMPTY'}")

# Check unique stones
all_stones = set()
for ep_id, ep in eps.items():
    for ex in ep.get('support', []) + ep.get('query', []):
        parts = ex.split(' -> ')
        for p in parts:
            m = re.search(r'\{color.*?\}', p)
            if m:
                all_stones.add(m.group(0))
print(f"\nTotal unique stones across all episodes: {len(all_stones)}")

# Now compare with the ORIGINAL full dataset
orig_train = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/shuffled_held_out_exps_generated_data_enhanced/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0.json"
with open(orig_train, 'r') as f:
    orig_data = json.load(f)

orig_eps = orig_data['episodes']
print(f"\nOriginal dataset episodes: {len(orig_eps)}")

# Check how many baseline episode IDs are in the original dataset
baseline_ids = set(ep_ids)
orig_ids = set(orig_eps.keys())
common = baseline_ids & orig_ids
print(f"Baseline episode IDs found in original: {len(common)} / {len(baseline_ids)}")

# For a few common episodes, compare the stone content
if common:
    sample_ids = list(common)[:3]
    for sid in sample_ids:
        b_support = eps[sid].get('support', [])
        o_support = orig_eps[sid].get('support', [])
        b_query = eps[sid].get('query', [])
        o_query = orig_eps[sid].get('query', [])
        
        print(f"\nEpisode {sid}:")
        print(f"  Baseline: {len(b_support)} support, {len(b_query)} query")
        print(f"  Original: {len(o_support)} support, {len(o_query)} query")
        
        if b_support and o_support:
            # Compare first support example
            if b_support[0] == o_support[0]:
                print(f"  First support: MATCH")
            else:
                print(f"  First support: MISMATCH!")
                print(f"    Baseline: {b_support[0][:120]}")
                print(f"    Original: {o_support[0][:120]}")
        
        # Check if all support examples match
        b_set = set(b_support)
        o_set = set(o_support)
        if b_set == o_set:
            print(f"  All support examples: MATCH")
        else:
            only_in_b = b_set - o_set
            only_in_o = o_set - b_set
            print(f"  Support MISMATCH! {len(only_in_b)} only in baseline, {len(only_in_o)} only in original")
            if only_in_b:
                print(f"    Example only in baseline: {list(only_in_b)[0][:120]}")
            if only_in_o:
                print(f"    Example only in original: {list(only_in_o)[0][:120]}")
