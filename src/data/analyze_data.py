import json
import os

# Load the JSON file from generated_data/ directory
json_files = ['/home/rsaha/projects/dm_alchemy/src/data/generated_data/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_2_qhop_1.json']
d = None
with open(json_files[0], 'r') as f:
    print(f"Loading JSON file: {json_files[0]}")
    d = json.load(f)
if d is None:
    raise ValueError("Failed to load JSON data.")
# Initialize variables to track min and max values
support_nums = []
query_nums = []

# Iterate over all episodes
for episode_id, episode_data in d['episodes'].items():
    support_nums.append(episode_data['support_num_generated'])
    query_nums.append(episode_data['query_num_generated'])

# Calculate min and max values
max_support_num = max(support_nums)
min_support_num = min(support_nums)
max_query_num = max(query_nums)
min_query_num = min(query_nums)

print(f"Support num generated - Min: {min_support_num}, Max: {max_support_num}")
print(f"Query num generated - Min: {min_query_num}, Max: {max_query_num}")