# The first key is the hop. The second key is data split seed, the third key is model init seed.
# Best runs from the hyperparameter tuning.
held_out_file_paths = {
        # Normalized reward paths.
        # 4: [
        #     "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/same_reward_held_out_color_4/all_graphs/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_9e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/predictions/",
        #     "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/same_reward_held_out_color_4/all_graphs/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_3/predictions/",
        #     # ""
        # ]
        
        4: {
            0: {
                42: "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/predictions/",
                0: "",
                1: "",
                2: "",
            },
            2: {
                42: "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_2/predictions/",
                0: "",
                1: "",
                3: "",
            },

            3: {
                42: "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/no_scheduler/wd_0.01_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_3/predictions/",
                0: "",
                1: "",
                2: "",
            }
        }
    }

# Frozen layer held-out jobs. The first key is the hop. The second key is data split seed, the third key is model init seed.
frozen_held_out_file_paths_per_layer_per_init_seed = {
    4: {
        0: {
            42: {
                'base_path': '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/',
                'freeze_epoch_200': {
                    'transformer_layer_0': ['/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/resume_from_epoch_200__freeze_transformer_layer_0/predictions/'],
                    'transformer_layer_1': [''],
                    'transformer_layer_2': [''],
                    'transformer_layer_3': [''],
                    'transformer_layer_4': [''],
                },
                'freeze_epoch_100': {
                    'transformer_layer_0': ['/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/resume_from_epoch_100__freeze_transformer_layer_0/predictions/'],
                    'transformer_layer_1': ['/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/resume_from_epoch_100__freeze_transformer_layer_1/predictions/'],
                    'transformer_layer_2': [''],
                    'transformer_layer_3': [''],
                    'transformer_layer_4': [''],
                },
                'freeze_epoch_110': {
                    'transformer_layer_0': [''],
                    'transformer_layer_1': [''],
                    'transformer_layer_2': [''],
                    'transformer_layer_3': [''],
                    'transformer_layer_4': [''],
                },
                'freeze_epoch_120': {
                    'transformer_layer_0': [''],
                    'transformer_layer_1': [''],
                    'transformer_layer_2': [''],
                    'transformer_layer_3': [''],
                    'transformer_layer_4': [''],
                },
                'freeze_epoch_130': {
                    'transformer_layer_0': [''],
                    'transformer_layer_1': [''],
                    'transformer_layer_2': [''],
                    'transformer_layer_3': [''],
                    'transformer_layer_4': [''],
                },
                'freeze_epoch_140': {
                    'transformer_layer_0': [''],
                    'transformer_layer_1': [''],
                    'transformer_layer_2': [''],
                    'transformer_layer_3': [''],
                    'transformer_layer_4': [''],
                },
                'freeze_epoch_150': {
                    'transformer_layer_0': ['/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/resume_from_epoch_150__freeze_transformer_layer_0/predictions/'],
                    'transformer_layer_1': [''],
                    'transformer_layer_2': [''],
                    'transformer_layer_3': [''],
                    'transformer_layer_4': [''],
                },
                'freeze_epoch_160': {
                    'transformer_layer_0': [''],
                    'transformer_layer_1': [''],
                    'transformer_layer_2': [''],
                    'transformer_layer_3': [''],
                    'transformer_layer_4': [''],
                },
                'freeze_epoch_170': {
                    'transformer_layer_0': [''],
                    'transformer_layer_1': [''],
                    'transformer_layer_2': [''],
                    'transformer_layer_3': [''],
                    'transformer_layer_4': [''],
                },
                'freeze_epoch_180': {
                    'transformer_layer_0': [''],
                    'transformer_layer_1': [''],
                    'transformer_layer_2': [''],
                    'transformer_layer_3': [''],
                    'transformer_layer_4': [''],
                },
                'freeze_epoch_190': {
                    'transformer_layer_0': [''],
                    'transformer_layer_1': [''],
                    'transformer_layer_2': [''],
                    'transformer_layer_3': [''],
                    'transformer_layer_4': [''],
                },
            },
            0: {
                'transformer_layer_0': [''],
                'transformer_layer_1': [''],
                'transformer_layer_2': [''],
                'transformer_layer_3': [''],
                'transformer_layer_4': [''],
            },
            1: {
                'transformer_layer_0': [''],
                'transformer_layer_1': [''],
                'transformer_layer_2': [''],
                'transformer_layer_3': [''],
                'transformer_layer_4': [''],
            },
            2: {
                'transformer_layer_0': [''],
                'transformer_layer_1': [''],
                'transformer_layer_2': [''],
                'transformer_layer_3': [''],
                'transformer_layer_4': [''],
            }
        },
    }
}


# Held out pickle files after analyzing predictions.
# The first key is data split seed, the second key is model init seed.
staged_accuracies_held_out_file_paths_baseline_pickles = {
    0: {
        42: "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/stagewise_accuracies_frozen_layer_hop_4_exp_held_out/stagewise_accuracies_data_split_seed_0_init_seed_42_hop_4_exp_held_out.pkl",
        0: "",
        1: "",
        2: "",
        3: ""
    },    
    1: {
        0: "",
        1: "",
        2: "",
        3: ""
    },
    2: {
        0: "",
        1: "",
        2: "",
        3: ""
    },

}
# The first key is data split seed, the second key is model init seed.
staged_accuracies_held_out_file_paths_frozen_pickles = {
    0: {
        42: {
            'base_path': '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/stagewise_accuracies_frozen_layer_hop_4_exp_held_out/',
            'freeze_epoch_200': {
                'transformer_layer_0': ['/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/stagewise_accuracies_frozen_layer_transformer_layer_0_data_split_seed_0_init_seed_42_hop_4_exp_held_out.pkl'],
                'transformer_layer_1': [''],
                'transformer_layer_2': [''],
                'transformer_layer_3': [''],
                'transformer_layer_4': [''],
            },
            'freeze_epoch_100': {
                'transformer_layer_0': ['/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/stagewise_accuracies_frozen_layer_transformer_layer_0_freeze_epoch_100_data_split_seed_0_init_seed_42_hop_4_exp_held_out.pkl'],
                'transformer_layer_1': ['/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/stagewise_accuracies_frozen_layer_transformer_layer_1_freeze_epoch_100_data_split_seed_0_init_seed_42_hop_4_exp_held_out.pkl'],
                'transformer_layer_2': [''],
                'transformer_layer_3': [''],
                'transformer_layer_4': [''],
            },
        },
        0: {
            'freeze_epoch_200': {
                'transformer_layer_0': [''],
                'transformer_layer_1': [''],
                'transformer_layer_2': [''],
                'transformer_layer_3': [''],
                'transformer_layer_4': [''],
            },
            'freeze_epoch_100': {
                'transformer_layer_0': [''],
                'transformer_layer_1': [''],
                'transformer_layer_2': [''],
                'transformer_layer_3': [''],
                'transformer_layer_4': [''],
            },
        },
        1: {
            'freeze_epoch_200': {
                'transformer_layer_0': [''],
                'transformer_layer_1': [''],
                'transformer_layer_2': [''],
                'transformer_layer_3': [''],
                'transformer_layer_4': [''],
            },
            'freeze_epoch_100': {
                'transformer_layer_0': [''],
                'transformer_layer_1': [''],
                'transformer_layer_2': [''],
                'transformer_layer_3': [''],
                'transformer_layer_4': [''],
            },
        },
        2: {
            'freeze_epoch_200': {
                'transformer_layer_0': [''],
                'transformer_layer_1': [''],
                'transformer_layer_2': [''],
                'transformer_layer_3': [''],
                'transformer_layer_4': [''],
            },
            'freeze_epoch_100': {
                'transformer_layer_0': [''],
                'transformer_layer_1': [''],
                'transformer_layer_2': [''],
                'transformer_layer_3': [''],
                'transformer_layer_4': [''],
            },
        },
    },
}








# #### Composition and Decomposition file paths for non-subsampled and subsampled data.


composition_file_paths_non_subsampled = {
    2: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_2_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_2_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_2_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
        ],
    
    3: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_3_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_3_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_3_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
        ],
    4: ['/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_4_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_4_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_4_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
        ],
    5: ['/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_5_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_5_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_5_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
        ]
}

composition_file_paths = {
    # 2: ['/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_step_lr/wd_0.001_lr_0.0001/step_size_165_gamma_0.2/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_0/predictions',
    #     '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_16/predictions', 
    #     '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_step_lr/wd_0.001_lr_0.0001/step_size_250_gamma_0.4/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_29/predictions'
    #     ],

    2: [
        # '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_0/predictions/',
        '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_0/flatten_linear_input/predictions/',
        '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_16/flatten_linear_input/predictions/',
        # '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_29/flatten_linear_input/predictions/',
        '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_29/flatten_linear_input/predictions/'
    ],
    3: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_3/seed_0/predictions', 
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_3/seed_16/predictions', 
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.1_lr_0.0001/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_3/seed_29/predictions'],

    4: ['/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_4/seed_0/predictions', 
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_4/seed_16/predictions', 
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.01_lr_0.0001/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_4/seed_29/predictions'],

    5: ['/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine_restarts/wd_0.001_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_5/seed_0/predictions',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.1_lr_0.0001/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_5/seed_16/predictions',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_5/seed_29/predictions']
}

decomposition_file_paths_non_subsampled = {
    2: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'

        # For anomaly runs of seed 29:
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
        ],

    3: [
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_3_classification_filter_True_input_features_output_stone_states_data.pkl', 
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl',
        # 'home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_4_classification_filter_True_input_features_output_stone_states_data.pkl'

        # Good runs from wandb.
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'

        # Anomalous runs to show the effect of bad hyperparameters.

        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
    ],
    4: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_4_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl', 
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_4_qhop_1_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_4_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
    ],


    5: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_5_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_5_qhop_1_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_5_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
        ]
}



decomposition_file_paths = {
    2: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_0/predictions/',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_16/predictions/',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_29/predictions/',


        # Anomalous runs to show the effect of bad hyperparameters.
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_29/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_29/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_29/predictions/'
        ],

        # NOTE: Why are there so many files for 3-hop decomposition? It's because we wanted to see if different hyperparameter settings made a difference in when the final phase was being learned and if there was overlap with other stages.

    3: [
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_0/predictions', 
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_3/predictions',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_3/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_0/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_3/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_0/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_29/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_4/predictions'

        # 3-hop anomalous runs to show the effect of bad hyperparameters.
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_29/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_29/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_29/predictions/'


        # Control runs:
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_9e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_0/predictions/'

        # Good runs from wandb.
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_0/predictions/',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_16/predictions/',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_29/predictions/',

        ],

    4: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_4_qhop_1/seed_0/predictions/',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_4_qhop_1/seed_16/predictions/',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_4_qhop_1/seed_29/predictions/'],

    5: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_0/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_2/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_16/predictions',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_16/predictions/',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_29/predictions/'
        
        
        
        # Don't use
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_16/predictions/',
        ]
}