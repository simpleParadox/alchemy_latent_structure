from DSA.dmd import DMD
from DSA.simdist import SimilarityTransformDist
import numpy as np
import torch

# Create a dictionary to store the dmd_analysis results for each epoch.
dmd_matrices = {}
layer_index = 0  # Example layer index to extract hidden states from.
epochs_to_save = [100, 250]
epoch1 = 100
epoch2 = 250
model_size = 'xsmall'
base_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/'
for epoch in epochs_to_save:
    # Load the model checkpoint for the given epoch.
    checkpoint_path = f"{base_path}/best_model_epoch_{epoch}_classification_{model_size}.pt"
    model_checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Extract the hidden states from the model checkpoint.
    # TODO: load the hidden states for a selected layer_index.
    hidden_states = model_checkpoint['model_state_dict']['classification_head.weight']
    import pdb; pdb.set_trace()

    dmd = DMD(hidden_states, n_delays=10, rank=100, device='cuda', verbose=True)

    dmd.fit()

    dmd_matrices[epoch] = dmd.A_v.cpu().numpy()


sim_dist = SimilarityTransformDist(device='cuda', iters=1000, lr=0.01)


score = sim_dist.fit_score(
    dmd_matrices[100],
    dmd_matrices[1000]
)