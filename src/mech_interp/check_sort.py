import sys; sys.path.append('src/models')
from data_loaders import AlchemyDataset

d = AlchemyDataset('src/data/chemistry_pickles/original_reward_potion_remap_generated_data/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_pairing_index_0.json', 
                   task_type='classification', filter_query_from_support=True, use_preprocessed=False, input_format='features', output_format='stone_states')
                   
print(f"Data type: {type(d.data)}, First item type: {type(d.data[0])}")
