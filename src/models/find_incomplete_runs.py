#!/usr/bin/env python3
"""
find_incomplete_runs.py

Scans prediction directories for frozen-layer training runs and identifies
incomplete or missing runs.  Generates a resubmission shell script containing
individual sbatch calls to rerun only the failed jobs.

Supports composition and decomposition experiment types.

Usage
-----
  python src/models/find_incomplete_runs.py \
      --exp_typ composition --hop 2 --data_split_seed 0

  # Dry-run: just print the report without writing the sbatch script
  python src/models/find_incomplete_runs.py \
      --exp_typ composition --hop 2 --data_split_seed 0 --dry_run
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Baseline checkpoint paths (extracted from baseline_and_frozen_filepaths.py)
# Key hierarchy:  exp_typ -> hop -> data_split_seed -> init_seed -> path
# The path is the PARENT directory (without trailing "/predictions/").
# ---------------------------------------------------------------------------

COMPOSITION_BASELINE_CHECKPOINT_PATHS: Dict[int, Dict[int, Dict[int, str]]] = {
    2: {
        0: {
            42: "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_0/init_seed_42/",
            1:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_8.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_0/init_seed_1/",
            3:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_0/init_seed_3/",
        }
    },
    3: {
        0: {
            42: "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_3/seed_0/init_seed_42/",
            1:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_3/seed_0/init_seed_1/",
            3:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_3/seed_0/init_seed_3/",
        }
    },
    4: {
        0: {
            42: "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_4/seed_0/init_seed_42/",
            1:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_9e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_4/seed_0/init_seed_1/",
            3:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_4/seed_0/init_seed_3/",
        }
    },
    5: {
        0: {
            42: "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_5/seed_0/init_seed_42/",
            1:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_8.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_5/seed_0/init_seed_1/",
            3:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_5/seed_0/init_seed_3/",
        }
    },
}

DECOMPOSITION_BASELINE_CHECKPOINT_PATHS: Dict[int, Dict[int, Dict[int, str]]] = {
    2: {
        0: {
            42: "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_0/init_seed_42/",
            1:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_0/init_seed_1/",
            3:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_8.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_0/init_seed_3/",
        }
    },
    3: {
        0: {
            42: "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_0/init_seed_42/",
            1:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_0/init_seed_1/",
            3:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_9e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_0/init_seed_3/",
        }
    },
    4: {
        0: {
            42: "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_4_qhop_1/seed_0/init_seed_42/",
            1:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_4_qhop_1/seed_0/init_seed_1/",
            3:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_4_qhop_1/seed_0/init_seed_3/",
        }
    },
    5: {
        0: {
            42: "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_0/init_seed_42/",
            2:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_9e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_0/init_seed_2/",
            3:  "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_0/init_seed_3/",
        }
    },
}


# ---------------------------------------------------------------------------
# Frozen epochs lookup — same data as delta_time_to_stage.py
# ---------------------------------------------------------------------------

COMPOSITION_FROZEN_EPOCHS: Dict[int, Dict[int, List[int]]] = {
    2: {
        42: [17, 27, 37, 47, 57, 67, 77, 87, 97, 107, 117, 127, 137, 147, 157, 167, 177, 187, 197, 207, 217, 227, 237, 247, 257, 267, 277, 287],
        1:  [12, 22, 32, 42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 232, 242, 252, 262],
        3:  [13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143, 153, 163, 173, 183],
    },
    3: {
        42: [13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143, 153, 163, 173],
        1:  [14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114, 124, 134, 144, 154, 164, 174, 184, 194, 204],
        3:  [11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221],
    },
    4: {
        42: [16, 26, 36, 46, 56, 66, 76, 86, 96, 106, 116, 126, 136, 146, 156, 166, 176, 186, 196, 206, 216, 226, 236, 246, 256],
        1:  [14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114, 124, 134, 144, 154, 164],
        3:  [15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175],
    },
    5: {
        42: [12, 22, 32, 42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222],
        1:  [15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175],
        3:  [13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143, 153, 163, 173, 183],
    },
}

DECOMPOSITION_FROZEN_EPOCHS: Dict[int, Dict[int, List[int]]] = {
    2: {
        42: [32, 42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 232, 242, 252, 262, 272, 282, 292, 302, 312, 322, 332, 342, 352, 362],
        1:  [48, 58, 68, 78, 88, 98, 108, 118, 128, 138, 148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 248, 258, 268, 278, 288, 298, 308, 318, 328, 338, 348, 358, 368, 378],
        3:  [32, 42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 232, 242, 252, 262, 272, 282, 292, 302, 312, 322, 332, 342, 352, 362, 372, 382, 392, 402, 412, 422, 432, 442, 452, 462, 472, 482, 492, 502, 512],
    },
    3: {
        42: [46, 56, 66, 76, 86, 96, 106, 116, 126, 136, 146, 156, 166, 176, 186, 196, 206, 216, 226, 236, 246, 256, 266, 276, 286, 296, 306, 316, 326, 336, 346, 356, 366, 376, 386, 396, 406, 416, 426, 436, 446, 456, 466, 476, 486, 496, 506, 516, 526, 536, 546, 556, 566],
        1:  [41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251, 261, 271, 281, 291, 301, 311, 321, 331, 341, 351, 361, 371, 381, 391, 401, 411, 421, 431, 441, 451, 461, 471],
        3:  [87, 97, 107, 117, 127, 137, 147, 157, 167, 177, 187, 197, 207, 217, 227, 237, 247, 257, 267, 277, 287, 297, 307, 317, 327, 337, 347, 357, 367, 377, 387, 397, 407, 417, 427, 437, 447, 457, 467],
    },
    4: {
        42: [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550],
        1:  [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550],
        3:  [62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 232, 242, 252, 262, 272, 282, 292, 302, 312, 322, 332, 342, 352, 362, 372, 382, 392, 402, 412, 422, 432, 442, 452, 462, 472, 482, 492],
    },
    5: {
        42: [124, 134, 144, 154, 164, 174, 184, 194, 204, 214, 224, 234, 244, 254, 264, 274, 284, 294, 304, 314, 324, 334, 344, 354, 364, 374, 384, 394, 404, 414, 424, 434, 444, 454, 464, 474, 484, 494, 504, 514, 524, 534, 544, 554, 564, 574, 584, 594, 604, 614, 624, 634, 644, 654, 664, 674, 684, 694, 704, 714, 724, 734],
        2:  [120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730],
        3:  [128, 138, 148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 248, 258, 268, 278, 288, 298, 308, 318, 328, 338, 348, 358, 368, 378, 388, 398, 408, 418, 428, 438, 448, 458, 468, 478, 488, 498, 508, 518, 528, 538, 548, 558, 568, 578, 588, 598, 608, 618, 628, 638, 648, 658, 668, 678, 688, 698, 708, 718, 728],
    },
}


# ---------------------------------------------------------------------------
# Data paths per experiment type + hop
# ---------------------------------------------------------------------------

COMPOSITION_DATA_PATHS: Dict[int, Dict[str, str]] = {
    2: {
        "train": "src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_generated_data/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_2.json",
        "val":   "src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_generated_data/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_2.json",
        "preprocessed": "src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_preprocessed",
    },
    3: {
        "train": "src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_generated_data/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_3.json",
        "val":   "src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_generated_data/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_3.json",
        "preprocessed": "src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_preprocessed",
    },
    4: {
        "train": "src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_generated_data/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_4.json",
        "val":   "src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_generated_data/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_4.json",
        "preprocessed": "src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_preprocessed",
    },
    5: {
        "train": "src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_generated_data/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_5.json",
        "val":   "src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_generated_data/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_5.json",
        "preprocessed": "src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_preprocessed",
    },
}

DECOMPOSITION_DATA_PATHS: Dict[int, Dict[str, str]] = {
    2: {
        "train": "src/data/generated_data_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1.json",
        "val":   "src/data/generated_data_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_2_qhop_1.json",
        "preprocessed": "src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes",
    },
    3: {
        "train": "src/data/generated_data_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1.json",
        "val":   "src/data/generated_data_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_3_qhop_1.json",
        "preprocessed": "src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes",
    },
    4: {
        "train": "src/data/generated_data_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_4_qhop_1.json",
        "val":   "src/data/generated_data_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_4_qhop_1.json",
        "preprocessed": "src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes",
    },
    5: {
        "train": "src/data/generated_data_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_5_qhop_1.json",
        "val":   "src/data/generated_data_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_5_qhop_1.json",
        "preprocessed": "src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes",
    },
}


# ---------------------------------------------------------------------------
# SLURM defaults (memory per hop, for multi-GPU jobs)
# ---------------------------------------------------------------------------
# Composition uses shop_1_qhop_N → shorter sequences → less memory.
# Decomposition uses shop_N_qhop_1 → longer sequences → more memory.
SLURM_MEM_COMPOSITION: Dict[int, str] = {2: "12G", 3: "12G", 4: "12G", 5: "12G"}
SLURM_MEM_DECOMPOSITION: Dict[int, str] = {2: "12G", 3: "12G", 4: "14G", 5: "14G"}

# Approximate seconds per epoch (used to compute dynamic SLURM --time)
SECONDS_PER_EPOCH_COMPOSITION = 10.7
SECONDS_PER_EPOCH_DECOMPOSITION = 25  # adjust if needed

STARTUP_OVERHEAD_SECONDS = 30  
TIME_BUFFER_FACTOR = 1.0       
MAX_SLURM_TIME_SECONDS = 24 * 3600  # cap at 24 hours


def compute_slurm_time(remaining_epochs: int, seconds_per_epoch: int) -> str:
    """Return an HH:MM:SS string sized to the remaining work."""
    raw = remaining_epochs * seconds_per_epoch
    total = int(raw * TIME_BUFFER_FACTOR) + STARTUP_OVERHEAD_SECONDS
    total = min(total, MAX_SLURM_TIME_SECONDS)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


FROZEN_LAYERS = [
    "embedding_layer",
    "transformer_layer_0",
    "transformer_layer_1",
    "transformer_layer_2",
    "transformer_layer_3",
]

TOTAL_EPOCHS = 1000  # All runs target 1000 epochs (indices 0..999)


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------

def count_prediction_files(pred_dir: str) -> int:
    """Count predictions_classification_epoch_*.npz files in a directory."""
    if not os.path.isdir(pred_dir):
        return 0
    return len(glob.glob(os.path.join(pred_dir, "predictions_classification_epoch_*.npz")))


def check_run_completeness(
    checkpoint_base_path: str,
    freeze_epoch: int,
    frozen_layer: str,
) -> Tuple[str, int, int, str]:
    """
    Check whether a single frozen-layer run is complete.

    Returns:
        (pred_dir, expected_files, actual_files, status)
        status is one of: 'complete', 'incomplete', 'missing'
    """
    # The frozen run saves predictions into:
    #   <checkpoint_base>/resume_from_epoch_<N>__freeze_<layer>/predictions/
    resume_subdir = f"resume_from_epoch_{freeze_epoch}__freeze_{frozen_layer}"
    pred_dir = os.path.join(checkpoint_base_path, resume_subdir, "predictions")

    # Expected: predictions from epoch (freeze_epoch+1) through epoch 999
    expected_files = TOTAL_EPOCHS - 1 - freeze_epoch  # epoch 0 has no prediction

    actual_files = count_prediction_files(pred_dir)

    if actual_files == 0 and not os.path.isdir(pred_dir):
        status = "missing"
    elif actual_files < expected_files:
        status = "incomplete"
    else:
        status = "complete"

    return (pred_dir, expected_files, actual_files, status)


def find_incomplete_runs(
    exp_typ: str,
    hop: int,
    data_split_seed: int,
    init_seeds: Optional[List[int]] = None,
) -> List[dict]:
    """
    Scan all expected (init_seed, frozen_layer, freeze_epoch) combinations
    and return a list of dicts for incomplete/missing runs.
    """
    if exp_typ == "composition":
        checkpoint_paths = COMPOSITION_BASELINE_CHECKPOINT_PATHS
        frozen_epochs_lookup = COMPOSITION_FROZEN_EPOCHS
    elif exp_typ == "decomposition":
        checkpoint_paths = DECOMPOSITION_BASELINE_CHECKPOINT_PATHS
        frozen_epochs_lookup = DECOMPOSITION_FROZEN_EPOCHS
    else:
        raise ValueError(f"Unsupported exp_typ: {exp_typ}")

    if hop not in checkpoint_paths:
        raise ValueError(f"hop={hop} not found in {exp_typ} checkpoint paths")
    if data_split_seed not in checkpoint_paths[hop]:
        raise ValueError(f"data_split_seed={data_split_seed} not found for {exp_typ} hop={hop}")
    if hop not in frozen_epochs_lookup:
        raise ValueError(f"hop={hop} not found in {exp_typ} frozen epochs lookup")

    seed_paths = checkpoint_paths[hop][data_split_seed]

    if init_seeds is None:
        init_seeds = sorted(seed_paths.keys())

    incomplete: List[dict] = []
    total_checked = 0
    total_complete = 0

    for init_seed in init_seeds:
        if init_seed not in seed_paths:
            print(f"WARNING: init_seed={init_seed} not in checkpoint paths for {exp_typ} hop={hop}. Skipping.")
            continue
        if init_seed not in frozen_epochs_lookup[hop]:
            print(f"WARNING: init_seed={init_seed} not in frozen epochs for {exp_typ} hop={hop}. Skipping.")
            continue

        base_path = seed_paths[init_seed].rstrip("/")
        freeze_epochs = frozen_epochs_lookup[hop][init_seed]

        for frozen_layer in FROZEN_LAYERS:
            for freeze_epoch in freeze_epochs:
                total_checked += 1
                pred_dir, expected, actual, status = check_run_completeness(
                    base_path, freeze_epoch, frozen_layer
                )

                if status == "complete":
                    total_complete += 1
                else:
                    incomplete.append({
                        "exp_typ": exp_typ,
                        "hop": hop,
                        "data_split_seed": data_split_seed,
                        "init_seed": init_seed,
                        "frozen_layer": frozen_layer,
                        "freeze_epoch": freeze_epoch,
                        "expected_files": expected,
                        "actual_files": actual,
                        "status": status,
                        "checkpoint_base_path": base_path,
                        "pred_dir": pred_dir,
                    })

    print(f"\n{'='*70}")
    print(f"SCAN SUMMARY: {exp_typ} {hop}-hop, data_split_seed={data_split_seed}")
    print(f"{'='*70}")
    print(f"  Total combinations checked: {total_checked}")
    print(f"  Complete:    {total_complete}")
    print(f"  Incomplete:  {sum(1 for r in incomplete if r['status'] == 'incomplete')}")
    print(f"  Missing:     {sum(1 for r in incomplete if r['status'] == 'missing')}")
    print(f"  Total to rerun: {len(incomplete)}")
    print(f"{'='*70}\n")

    return incomplete


# ---------------------------------------------------------------------------
# Report and sbatch generation
# ---------------------------------------------------------------------------

def print_report(incomplete_runs: List[dict]) -> None:
    """Pretty-print the incomplete runs as a table."""
    if not incomplete_runs:
        print("All runs are complete! Nothing to resubmit.")
        return

    # Header
    fmt = "{:<12} {:<25} {:<14} {:<10} {:<10} {:<10}"
    print(fmt.format("init_seed", "frozen_layer", "freeze_epoch", "expected", "actual", "status"))
    print("-" * 85)
    for r in incomplete_runs:
        print(fmt.format(
            r["init_seed"],
            r["frozen_layer"],
            r["freeze_epoch"],
            r["expected_files"],
            r["actual_files"],
            r["status"],
        ))


def generate_sbatch_scripts(
    incomplete_runs: List[dict],
    output_path: str,
    exp_typ: str,
    hop: int,
    account: str = "aip-afyshe",
) -> None:
    """
    Generate one standalone sbatch file per incomplete run, stored in a
    dedicated directory: reruns_sbatch_<exp_typ>_<hop>hop/

    Also generates a thin master script (output_path) that simply submits
    all individual sbatch files via ``sbatch``.
    """
    if not incomplete_runs:
        print("No incomplete runs — not generating sbatch scripts.")
        return

    if exp_typ == "composition":
        data_paths = COMPOSITION_DATA_PATHS[hop]
        mem_lookup = SLURM_MEM_COMPOSITION
    elif exp_typ == "decomposition":
        data_paths = DECOMPOSITION_DATA_PATHS[hop]
        mem_lookup = SLURM_MEM_DECOMPOSITION
    else:
        raise ValueError(f"Unsupported exp_typ: {exp_typ}")

    mem = mem_lookup.get(hop, "14G")
    sec_per_epoch = (
        SECONDS_PER_EPOCH_COMPOSITION if exp_typ == "composition"
        else SECONDS_PER_EPOCH_DECOMPOSITION
    )

    # Directory that holds individual sbatch scripts
    sbatch_dir = f"reruns_sbatch_{exp_typ}_{hop}hop"
    os.makedirs(sbatch_dir, exist_ok=True)

    # Directory for SLURM stdout logs
    log_dir = f"reruns_{exp_typ}/{hop}-hop"
    os.makedirs(log_dir, exist_ok=True)

    generated_files: List[str] = []

    for r in incomplete_runs:
        init_seed = r["init_seed"]
        frozen_layer = r["frozen_layer"]
        freeze_epoch = r["freeze_epoch"]
        checkpoint_path = r["checkpoint_base_path"]

        # Compute dynamic time allocation based on remaining epochs
        remaining_epochs = TOTAL_EPOCHS - freeze_epoch
        time_limit = compute_slurm_time(remaining_epochs, sec_per_epoch)

        job_tag = f"{exp_typ}_{hop}hop_s{init_seed}_{frozen_layer}_e{freeze_epoch}"
        sbatch_filename = f"rerun_{job_tag}.sh"
        sbatch_filepath = os.path.join(sbatch_dir, sbatch_filename)

        # Build the accelerate launch command
        train_cmd = (
            f"CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --gpu_ids=all --main_process_port=0 "
            f"src/models/train.py "
            f"--resume_from_checkpoint=True "
            f"--resume_checkpoint_epoch={freeze_epoch} "
            f"--resume_checkpoint_path={checkpoint_path} "
            f"--freeze_layers={frozen_layer} "
            f"--epochs=1000 "
            f"--batch_size=32 "
            f"--model_size=xsmall "
            f"--model_architecture=decoder "
            f"--task_type=classification "
            f"--input_format=features "
            f"--output_format=stone_states "
            f"--override_num_classes=108 "
            f"--pooling_strategy=last_token "
            f"--padding_side=right "
            f"--use_flash_attention=True "
            f"--use_preprocessed=True "
            f"--save_checkpoints=True "
            f"--store_predictions=True "
            f"--use_scheduler=True "
            f"--scheduler_type=cosine "
            f"--scheduler_call_location=after_batch "
            f"--t0=20 "
            f"--wandb_mode=online "
            f"--wandb_project=alchemy-meta-learning "
            f"--log_interval=50 "
            f"--num_workers=4 "
            f"--fp16=False "
            f"--use_truncation=False "
            f"--allow_data_path_mismatch=False "
            f"--custom_checkpoint_dir=aip-afyshe "
            f"--filter_query_from_support=True "
            f"--include_nonlinearity=True "
            f"--flatten_linear_model_input=False "
            f"--store_in_scratch=False "
            f"--multi_gpu_validation=False "
            f"--max_seq_len=2048 "
            f"--reduce_factor=0.2 "
            f"--reduce_patience=10 "
            f"--step_size=165 "
            f"--val_split_seed=42 "
            f"--prediction_type=default "
            f"--multi_label_reduction=mean "
            f"--train_data_path={data_paths['train']} "
            f"--val_data_path={data_paths['val']} "
            f"--preprocessed_dir={data_paths['preprocessed']}"
        )

        # Write the individual sbatch file
        sbatch_lines = [
            "#!/bin/bash",
            f"#SBATCH --account={account}",
            "#SBATCH --mail-user=rsaha@ualberta.ca",
            "#SBATCH --mail-type=ALL",
            "#SBATCH --gres=gpu:4",
            "#SBATCH --cpus-per-task=4",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --job-name=rerun_{job_tag}",
            f"#SBATCH --output={log_dir}/rerun_{job_tag}-%j.out",
            "",
            "module load python/3.10",
            "source alchemy_env/bin/activate",
            "",
            "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "WANDB__SERVICE_WAIT=300",
            "WANDB_INIT_TIMEOUT=300",
            "export WANDB__SERVICE_WAIT",
            "export WANDB_INIT_TIMEOUT",
            "",
            train_cmd,
            "",
        ]

        with open(sbatch_filepath, "w") as f:
            f.write("\n".join(sbatch_lines))
        os.chmod(sbatch_filepath, 0o755)
        generated_files.append(sbatch_filepath)

    # Write the master submission script
    master_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Master resubmission script for {exp_typ} {hop}-hop incomplete runs",
        f"# Total jobs: {len(incomplete_runs)}",
        f"# Individual sbatch scripts are in: {sbatch_dir}/",
        "",
    ]
    for filepath in generated_files:
        master_lines.append(f"sbatch {filepath}")
    master_lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(master_lines))
    os.chmod(output_path, 0o755)

    print(f"\nGenerated {len(generated_files)} individual sbatch scripts in: {sbatch_dir}/")
    print(f"Master submission script written to: {output_path}")
    print(f"  Run with:  bash {output_path}")
    print(f"  Or submit individually:  sbatch {sbatch_dir}/rerun_<job_tag>.sh")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect incomplete frozen-layer training runs and generate SLURM resubmission scripts."
    )
    p.add_argument("--exp_typ", type=str, required=True, choices=["composition", "decomposition"],
                    help="Experiment type.")
    p.add_argument("--hop", type=int, required=True,
                    help="Hop length (e.g. 2, 3, 4, 5).")
    p.add_argument("--data_split_seed", type=int, default=0,
                    help="Data split seed (default: 0).")
    p.add_argument("--init_seeds", type=int, nargs="+", default=None,
                    help="Init seeds to check. Default: all seeds for the given (exp_typ, hop).")
    p.add_argument("--output", type=str, default=None,
                    help="Path for the resubmission script. Default: resubmit_<exp_typ>_<hop>hop.sh")
    p.add_argument("--account", type=str, default="aip-afyshe",
                    help="SLURM account to use (default: aip-afyshe).")
    p.add_argument("--dry_run", action="store_true",
                    help="Only print the report, do not write the sbatch script.")

    return p.parse_args()


def main():
    args = parse_args()

    if args.output is None:
        args.output = f"resubmit_{args.exp_typ}_{args.hop}hop.sh"

    incomplete = find_incomplete_runs(
        exp_typ=args.exp_typ,
        hop=args.hop,
        data_split_seed=args.data_split_seed,
        init_seeds=args.init_seeds,
    )

    print_report(incomplete)

    if args.dry_run:
        print("\n[DRY RUN] Not writing sbatch script.")
    else:
        generate_sbatch_scripts(
            incomplete_runs=incomplete,
            output_path=args.output,
            exp_typ=args.exp_typ,
            hop=args.hop,
            account=args.account,
        )


if __name__ == "__main__":
    main()
