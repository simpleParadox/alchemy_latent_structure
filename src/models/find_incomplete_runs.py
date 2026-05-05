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
import csv
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from cluster_profile import cluster

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
        3:  [32, 42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 232, 242, 252, 262, 272, 282, 292, 302, 312, 322, 332, 342, 352, 362, 372, 382, 392, 402, 412, 422, 432, 442, 452, 462, 472, 482, 492, 502, 512]
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
        42:   [124, 134, 144, 154, 164, 174, 184, 194, 204, 214, 224, 234, 244, 254, 264, 274, 284, 294, 304, 314, 324, 334, 344, 354, 364, 374, 384, 394, 404, 414, 424, 434, 444, 454, 464, 474, 484, 494, 504, 514, 524, 534, 544, 554, 564, 574, 584, 594, 604, 614, 624, 634, 644, 654, 664, 674, 684, 694, 704, 714, 724, 734, 744],
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
SLURM_MEM_DECOMPOSITION: Dict[int, str] = {2: "12G", 3: "14G", 4: "15G", 5: "12G"}

# Approximate seconds per epoch (used to compute dynamic SLURM --time)
SECONDS_PER_EPOCH_COMPOSITION = {
    2: 11,
    3: 11,
    4: 11,
    5: 11,
}
SECONDS_PER_EPOCH_DECOMPOSITION =  {
    2: 11,
    3: 33,
    4: 61,
    5: 150
}

STARTUP_OVERHEAD_SECONDS = 60
TIME_BUFFER_FACTOR = 1.0
MAX_SLURM_TIME_SECONDS = 50 * 3600  # cap at 24 hours


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
DEFAULT_MAX_MISSING_EPOCHS = 20
DEFAULT_METRIC_KEY = "predicted_exact_out_of_all_108"
DEFAULT_METRIC_THRESHOLD = 0.975
DEFAULT_METRIC_PATIENCE = 10


def extract_scheduler_type(checkpoint_path: str) -> str:
    """Extract the scheduler type from the checkpoint path.

    Paths contain segments like ``scheduler_cosine/`` or
    ``scheduler_cosine_restarts/``.  We match the most specific
    variant first to avoid returning ``cosine`` for
    ``cosine_restarts``.
    """
    if "scheduler_cosine_restarts" in checkpoint_path:
        return "cosine_restarts"
    elif "scheduler_cosine" in checkpoint_path:
        return "cosine"
    elif "no_scheduler" in checkpoint_path:
        return "none"
    else:
        import re
        m = re.search(r'scheduler_([\w]+)', checkpoint_path)
        if m:
            return m.group(1)
        return "cosine"  # safe default


def checkpoint_base_to_rel(checkpoint_base_path: str) -> str:
    """Convert absolute checkpoint base path to path relative to src/saved_models."""
    normalized = checkpoint_base_path.replace("\\", "/")
    marker = "/src/saved_models/"
    idx = normalized.find(marker)
    if idx == -1:
        raise ValueError(
            f"Could not derive checkpoint_base_rel from path (missing '{marker}'): {checkpoint_base_path}"
        )
    rel = normalized[idx + len(marker):].strip("/")
    return rel


def default_saved_models_root_for_cluster(cluster_name: str) -> str:
    """Return default src/saved_models root for the current cluster."""
    if cluster_name == "cc":
        return "/home/rsaha/scratch/dm_alchemy/src/saved_models"
    # killarney and other local clusters use the aip project tree by default.
    return "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models"


def resolve_saved_models_root(saved_models_root_override: Optional[str] = None) -> str:
    if saved_models_root_override:
        return saved_models_root_override.rstrip("/")
    return default_saved_models_root_for_cluster(cluster)


def resolve_checkpoint_base_path(path_or_rel: str, saved_models_root: str) -> str:
    """
    Resolve a checkpoint base path for the active cluster.

    Accepts either:
      - absolute checkpoint path containing /src/saved_models/
      - relative checkpoint path under src/saved_models
    """
    if os.path.isabs(path_or_rel):
        rel = checkpoint_base_to_rel(path_or_rel)
    else:
        rel = path_or_rel.strip("/").replace("\\", "/")
    return os.path.join(saved_models_root, rel).rstrip("/")


def _extract_epoch_from_predictions_filename(path: str) -> Optional[int]:
    m = re.search(r"predictions_classification_epoch_(\d+)\.npz$", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def _load_targets_from_baseline_predictions_dir(checkpoint_base_path: str) -> Tuple[Optional[np.ndarray], str, str]:
    """
    Load baseline validation targets from <checkpoint_base_path>/predictions/.

    Returns:
      (targets_array_or_none, reason, targets_path_used)
    """
    baseline_predictions_dir = os.path.join(checkpoint_base_path, "predictions")
    candidates = [
        os.path.join(baseline_predictions_dir, "targets_classification_epoch_001.npz"),
        os.path.join(baseline_predictions_dir, "targets_classification_epoch_002.npz"),
    ]

    chosen_path = ""
    for candidate in candidates:
        if os.path.exists(candidate):
            chosen_path = candidate
            break

    if not chosen_path:
        return None, "targets_missing", ""

    try:
        loaded = np.load(chosen_path, allow_pickle=True)
    except Exception:
        return None, "targets_unreadable", chosen_path

    if "targets" not in loaded:
        return None, "targets_missing_key", chosen_path

    targets = np.asarray(loaded["targets"])
    targets = targets.reshape(-1)
    return targets, "ok", chosen_path


def _prediction_to_class_ids(predictions: np.ndarray) -> np.ndarray:
    arr = np.asarray(predictions)
    if arr.ndim > 1:
        arr = np.argmax(arr, axis=-1)
    return arr.reshape(-1)


def _collect_epoch_accuracies_from_predictions(
    pred_dir: str,
    targets: np.ndarray,
) -> Tuple[List[Tuple[int, float]], Dict[str, int]]:
    diagnostics: Dict[str, int] = {
        "prediction_files_missing": 0,
        "prediction_file_unreadable": 0,
        "prediction_missing_key": 0,
        "prediction_shape_mismatch": 0,
        "no_valid_prediction_epochs": 0,
    }

    pattern = os.path.join(pred_dir, "predictions_classification_epoch_*.npz")
    pred_files = glob.glob(pattern)
    if not pred_files:
        diagnostics["prediction_files_missing"] += 1
        diagnostics["no_valid_prediction_epochs"] += 1
        return [], diagnostics

    rows: List[Tuple[int, float]] = []
    for path in pred_files:
        epoch_num = _extract_epoch_from_predictions_filename(path)
        if epoch_num is None:
            continue

        try:
            loaded = np.load(path, allow_pickle=True)
        except Exception:
            diagnostics["prediction_file_unreadable"] += 1
            continue

        if "predictions" not in loaded:
            diagnostics["prediction_missing_key"] += 1
            continue

        pred_ids = _prediction_to_class_ids(loaded["predictions"])
        if pred_ids.shape[0] != targets.shape[0]:
            diagnostics["prediction_shape_mismatch"] += 1
            continue

        acc = float(np.mean(pred_ids == targets))
        rows.append((epoch_num, acc))

    rows.sort(key=lambda x: x[0])
    if not rows:
        diagnostics["no_valid_prediction_epochs"] += 1
    return rows, diagnostics


def _metric_crossed_threshold_with_streak(
    epoch_values: List[Tuple[int, float]],
    threshold: float,
    patience: int,
) -> Tuple[bool, int]:
    streak = 0
    max_streak = 0
    prev_epoch: Optional[int] = None
    for epoch, val in epoch_values:
        if val >= threshold:
            if prev_epoch is not None and epoch == prev_epoch + 1:
                streak += 1
            else:
                streak = 1
            if streak > max_streak:
                max_streak = streak
            if streak >= patience:
                return True, max_streak
        else:
            streak = 0
        prev_epoch = epoch
    return False, max_streak


def apply_metric_threshold_gate(
    runs: List[dict],
    metric_threshold: float,
    metric_patience: int,
    metric_key: str = DEFAULT_METRIC_KEY,
) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    """
    Filter only status='incomplete' runs using metric threshold streak checks.

    Returns:
      (runs_after_gate, threshold_met_runs, diagnostics)
    """
    remaining_runs: List[dict] = []
    threshold_met_runs: List[dict] = []

    diagnostics: Dict[str, int] = {
        "eligible_incomplete_runs": 0,
        "threshold_crossed": 0,
        "threshold_not_crossed": 0,
        "targets_missing": 0,
        "targets_unreadable": 0,
        "targets_missing_key": 0,
        "prediction_files_missing": 0,
        "prediction_file_unreadable": 0,
        "prediction_missing_key": 0,
        "prediction_shape_mismatch": 0,
        "no_valid_prediction_epochs": 0,
    }

    for run in runs:
        if run.get("status") != "incomplete":
            remaining_runs.append(run)
            continue

        diagnostics["eligible_incomplete_runs"] += 1

        targets, targets_reason, targets_path = _load_targets_from_baseline_predictions_dir(
            checkpoint_base_path=run["checkpoint_base_path"],
        )

        run["metric_key"] = metric_key
        run["metric_threshold"] = metric_threshold
        run["metric_patience"] = metric_patience
        run["targets_path"] = targets_path

        if targets is None:
            run["metric_check_reason"] = targets_reason
            if targets_reason in diagnostics:
                diagnostics[targets_reason] += 1
            diagnostics["threshold_not_crossed"] += 1
            remaining_runs.append(run)
            continue

        epoch_values, pred_diag = _collect_epoch_accuracies_from_predictions(
            pred_dir=run["pred_dir"],
            targets=targets,
        )
        for k, v in pred_diag.items():
            diagnostics[k] += v

        run["metric_points"] = len(epoch_values)
        run["metric_last_value"] = epoch_values[-1][1] if epoch_values else None
        run["metric_last_epoch"] = epoch_values[-1][0] if epoch_values else None

        if not epoch_values:
            run["metric_check_reason"] = "no_valid_prediction_epochs"
            diagnostics["threshold_not_crossed"] += 1
            remaining_runs.append(run)
            continue

        crossed, max_streak = _metric_crossed_threshold_with_streak(
            epoch_values=epoch_values,
            threshold=metric_threshold,
            patience=metric_patience,
        )
        run["metric_max_streak"] = max_streak
        run["metric_check_reason"] = "ok"

        if crossed:
            run["status"] = "threshold_met"
            diagnostics["threshold_crossed"] += 1
            threshold_met_runs.append(run)
        else:
            diagnostics["threshold_not_crossed"] += 1
            remaining_runs.append(run)

    return remaining_runs, threshold_met_runs, diagnostics


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
    max_missing_epochs: int = DEFAULT_MAX_MISSING_EPOCHS,
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
    # print(pred_dir)

    # Expected: predictions from epoch (freeze_epoch+1) through epoch 999
    expected_files = TOTAL_EPOCHS - 1 - freeze_epoch  # epoch 0 has no prediction

    actual_files = count_prediction_files(pred_dir)

    if actual_files == 0 and not os.path.isdir(pred_dir):
        status = "missing"
    elif (expected_files - actual_files) > max_missing_epochs:
        status = "incomplete"
        print(pred_dir)
    else:
        status = "complete"

    return (pred_dir, expected_files, actual_files, status)


def find_incomplete_runs(
    exp_typ: str,
    hop: int,
    data_split_seed: int,
    init_seeds: Optional[List[int]] = None,
    max_missing_epochs: int = DEFAULT_MAX_MISSING_EPOCHS,
    saved_models_root_override: Optional[str] = None,
    enable_metric_threshold_gate: bool = False,
    metric_threshold: float = DEFAULT_METRIC_THRESHOLD,
    metric_patience: int = DEFAULT_METRIC_PATIENCE,
) -> Tuple[List[dict], List[dict], List[dict], Dict[str, int]]:
    """
        Scan all expected (init_seed, frozen_layer, freeze_epoch) combinations
        and return two lists:
            - incomplete/missing runs
            - complete runs
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

    saved_models_root = resolve_saved_models_root(saved_models_root_override)

    incomplete: List[dict] = []
    complete_runs: List[dict] = []
    total_checked = 0
    total_complete = 0

    for init_seed in init_seeds:
        if init_seed not in seed_paths:
            print(f"WARNING: init_seed={init_seed} not in checkpoint paths for {exp_typ} hop={hop}. Skipping.")
            continue
        if init_seed not in frozen_epochs_lookup[hop]:
            print(f"WARNING: init_seed={init_seed} not in frozen epochs for {exp_typ} hop={hop}. Skipping.")
            continue

        base_path = resolve_checkpoint_base_path(seed_paths[init_seed], saved_models_root)
        freeze_epochs = frozen_epochs_lookup[hop][init_seed]

        for frozen_layer in FROZEN_LAYERS:
            for freeze_epoch in freeze_epochs:
                total_checked += 1
                pred_dir, expected, actual, status = check_run_completeness(
                    base_path,
                    freeze_epoch,
                    frozen_layer,
                    max_missing_epochs=max_missing_epochs,
                )

                if status == "complete":
                    total_complete += 1
                    complete_runs.append({
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

    threshold_met_runs: List[dict] = []
    metric_gate_diagnostics: Dict[str, int] = {
        "eligible_incomplete_runs": 0,
        "threshold_crossed": 0,
        "threshold_not_crossed": 0,
        "targets_missing": 0,
        "targets_unreadable": 0,
        "targets_missing_key": 0,
        "prediction_files_missing": 0,
        "prediction_file_unreadable": 0,
        "prediction_missing_key": 0,
        "prediction_shape_mismatch": 0,
        "no_valid_prediction_epochs": 0,
    }

    if enable_metric_threshold_gate:
        incomplete, threshold_met_runs, metric_gate_diagnostics = apply_metric_threshold_gate(
            runs=incomplete,
            metric_threshold=metric_threshold,
            metric_patience=metric_patience,
            metric_key=DEFAULT_METRIC_KEY,
        )

        print(f"{'='*70}")
        print("METRIC GATE SUMMARY (from predictions .npz, applied to status='incomplete' only)")
        print(f"{'='*70}")
        print(f"  Metric key:        {DEFAULT_METRIC_KEY}")
        print(f"  Threshold:         {metric_threshold}")
        print(f"  Patience (streak): {metric_patience}")
        print(f"  Eligible runs:     {metric_gate_diagnostics['eligible_incomplete_runs']}")
        print(f"  Threshold crossed: {metric_gate_diagnostics['threshold_crossed']}")
        print(f"  Still rerun:       {metric_gate_diagnostics['threshold_not_crossed']}")

        warning_keys = [
            "targets_missing",
            "targets_unreadable",
            "targets_missing_key",
            "prediction_files_missing",
            "prediction_file_unreadable",
            "prediction_missing_key",
            "prediction_shape_mismatch",
            "no_valid_prediction_epochs",
        ]
        warning_total = sum(metric_gate_diagnostics[k] for k in warning_keys)
        if warning_total > 0:
            print("  Warnings (treated as not crossed):")
            for k in warning_keys:
                if metric_gate_diagnostics[k] > 0:
                    print(f"    - {k}: {metric_gate_diagnostics[k]}")
        print(f"{'='*70}\n")

    return incomplete, complete_runs, threshold_met_runs, metric_gate_diagnostics


# ---------------------------------------------------------------------------
# Manual runs loading
# ---------------------------------------------------------------------------

def load_manual_runs(
    manual_runs_path: str,
    exp_typ: str,
    hop: int,
    data_split_seed: int,
    saved_models_root_override: Optional[str] = None,
) -> List[dict]:
    """
    Load a CSV file of explicit (init_seed, frozen_layer, freeze_epoch)
    tuples and build the same list-of-dicts structure used by
    generate_sbatch_scripts.

    File format — one run per line, comma-separated:
        init_seed,frozen_layer,freeze_epoch

    Example lines:
        42,transformer_layer_0,430
        42,embedding_layer,480
        1,transformer_layer_3,470

    Lines starting with '#' and blank lines are ignored.
    """
    if exp_typ == "composition":
        checkpoint_paths = COMPOSITION_BASELINE_CHECKPOINT_PATHS
    elif exp_typ == "decomposition":
        checkpoint_paths = DECOMPOSITION_BASELINE_CHECKPOINT_PATHS
    else:
        raise ValueError(f"Unsupported exp_typ: {exp_typ}")

    seed_paths = checkpoint_paths[hop][data_split_seed]

    saved_models_root = resolve_saved_models_root(saved_models_root_override)

    runs: List[dict] = []
    with open(manual_runs_path, "r", newline="") as f:
        raw_lines = [ln.rstrip("\n") for ln in f]

    data_lines = [ln for ln in raw_lines if ln.strip() and not ln.lstrip().startswith("#")]
    if not data_lines:
        print(f"\nLoaded 0 manual run(s) from: {manual_runs_path}")
        return runs

    # Detect rich/header format vs legacy 3-column format.
    first_cols = [c.strip() for c in data_lines[0].split(",")]
    has_header = len(first_cols) >= 3 and first_cols[0] in {
        "init_seed", "exp_typ", "hop", "data_split_seed", "frozen_layer", "freeze_epoch"
    }

    if has_header:
        reader = csv.DictReader(data_lines)
        for row_idx, row in enumerate(reader, start=2):
            try:
                init_seed = int(str(row.get("init_seed", "")).strip())
                frozen_layer = str(row.get("frozen_layer", "")).strip()
                freeze_epoch = int(str(row.get("freeze_epoch", "")).strip())
            except Exception:
                print(f"WARNING: skipping malformed CSV row {row_idx}: {row}")
                continue

            checkpoint_base_rel = str(row.get("checkpoint_base_rel", "")).strip()
            checkpoint_base_abs = str(row.get("checkpoint_base_path", "")).strip()

            if checkpoint_base_rel:
                base_path = resolve_checkpoint_base_path(checkpoint_base_rel, saved_models_root)
            elif checkpoint_base_abs:
                base_path = resolve_checkpoint_base_path(checkpoint_base_abs, saved_models_root)
            else:
                if init_seed not in seed_paths:
                    print(
                        f"WARNING: init_seed={init_seed} not in checkpoint paths for {exp_typ} hop={hop}. "
                        f"Skipping row {row_idx}."
                    )
                    continue
                base_path = resolve_checkpoint_base_path(seed_paths[init_seed], saved_models_root)

            resume_subdir = f"resume_from_epoch_{freeze_epoch}__freeze_{frozen_layer}"
            pred_dir = os.path.join(base_path, resume_subdir, "predictions")

            runs.append({
                "exp_typ": exp_typ,
                "hop": hop,
                "data_split_seed": data_split_seed,
                "init_seed": init_seed,
                "frozen_layer": frozen_layer,
                "freeze_epoch": freeze_epoch,
                "expected_files": TOTAL_EPOCHS - 1 - freeze_epoch,
                "actual_files": "N/A",
                "status": "manual",
                "checkpoint_base_path": base_path,
                "pred_dir": pred_dir,
            })
    else:
        # Legacy format: init_seed,frozen_layer,freeze_epoch
        for line_no, raw_line in enumerate(raw_lines, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                print(f"WARNING: skipping malformed line {line_no}: {raw_line.rstrip()}")
                continue

            init_seed = int(parts[0])
            frozen_layer = parts[1]
            freeze_epoch = int(parts[2])

            if init_seed not in seed_paths:
                print(
                    f"WARNING: init_seed={init_seed} not in checkpoint paths "
                    f"for {exp_typ} hop={hop}. Skipping line {line_no}."
                )
                continue

            base_path = resolve_checkpoint_base_path(seed_paths[init_seed], saved_models_root)
            resume_subdir = f"resume_from_epoch_{freeze_epoch}__freeze_{frozen_layer}"
            pred_dir = os.path.join(base_path, resume_subdir, "predictions")

            runs.append({
                "exp_typ": exp_typ,
                "hop": hop,
                "data_split_seed": data_split_seed,
                "init_seed": init_seed,
                "frozen_layer": frozen_layer,
                "freeze_epoch": freeze_epoch,
                "expected_files": TOTAL_EPOCHS - 1 - freeze_epoch,
                "actual_files": "N/A",
                "status": "manual",
                "checkpoint_base_path": base_path,
                "pred_dir": pred_dir,
            })

    print(f"\nLoaded {len(runs)} manual run(s) from: {manual_runs_path}")
    return runs


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


def print_complete_report(complete_runs: List[dict]) -> None:
    """Pretty-print only completed runs, with pred_dir listed after the table."""
    if not complete_runs:
        print("No completed runs found.")
        return

    fmt = "{:<12} {:<25} {:<14} {:<10} {:<10} {:<10}"
    print(fmt.format("init_seed", "frozen_layer", "freeze_epoch", "expected", "actual", "status"))
    print("-" * 85)
    for r in complete_runs:
        print(fmt.format(
            r["init_seed"],
            r["frozen_layer"],
            r["freeze_epoch"],
            r["expected_files"],
            r["actual_files"],
            r["status"],
        ))

    print("\nPrediction directories (completed runs):")
    for r in complete_runs:
        print(r["pred_dir"])


def print_threshold_met_report(threshold_met_runs: List[dict]) -> None:
    """Pretty-print runs excluded by metric threshold gate."""
    if not threshold_met_runs:
        print("No runs were excluded by metric threshold gate.")
        return

    fmt = "{:<12} {:<25} {:<14} {:<12} {:<10} {:<10} {:<10}"
    print("Runs excluded from rerun because metric threshold was crossed:")
    print(fmt.format("init_seed", "frozen_layer", "freeze_epoch", "max_streak", "points", "last_val", "last_ep"))
    print("-" * 112)
    for r in threshold_met_runs:
        last_val = r.get("metric_last_value")
        last_val_str = f"{last_val:.4f}" if isinstance(last_val, float) else "N/A"
        print(fmt.format(
            r["init_seed"],
            r["frozen_layer"],
            r["freeze_epoch"],
            r.get("metric_max_streak", "N/A"),
            r.get("metric_points", "N/A"),
            last_val_str,
            r.get("metric_last_epoch", "N/A"),
        ))


def export_manual_runs_csv(runs: List[dict], output_csv_path: str) -> None:
    """Export runs into a portable CSV suitable for --manual_runs."""
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)

    fieldnames = [
        "exp_typ",
        "hop",
        "data_split_seed",
        "init_seed",
        "frozen_layer",
        "freeze_epoch",
        "checkpoint_base_rel",
        "checkpoint_base_path",
        "status",
        "expected_files",
        "actual_files",
        "pred_dir",
    ]

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs:
            try:
                checkpoint_base_rel = checkpoint_base_to_rel(r["checkpoint_base_path"])
            except ValueError:
                checkpoint_base_rel = ""

            writer.writerow({
                "exp_typ": r.get("exp_typ", ""),
                "hop": r.get("hop", ""),
                "data_split_seed": r.get("data_split_seed", ""),
                "init_seed": r.get("init_seed", ""),
                "frozen_layer": r.get("frozen_layer", ""),
                "freeze_epoch": r.get("freeze_epoch", ""),
                "checkpoint_base_rel": checkpoint_base_rel,
                "checkpoint_base_path": r.get("checkpoint_base_path", ""),
                "status": r.get("status", ""),
                "expected_files": r.get("expected_files", ""),
                "actual_files": r.get("actual_files", ""),
                "pred_dir": r.get("pred_dir", ""),
            })

    print(f"Exported {len(runs)} run(s) to CSV: {output_csv_path}")


def generate_sbatch_scripts(
    incomplete_runs: List[dict],
    output_path: str,
    exp_typ: str,
    hop: int,
    account: str = "aip-afyshe",
    enable_chunked_resume: bool = False,
    chunk_epochs: int = 200,
    chunk_checkpoint_every: int = 5,
    chunk_keep_last_n: int = 5,
    auto_chain_afterok: bool = False,
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
        SECONDS_PER_EPOCH_COMPOSITION[hop] if exp_typ == "composition"
        else SECONDS_PER_EPOCH_DECOMPOSITION[hop]
    )

    # Directory that holds individual sbatch scripts
    sbatch_dir = f"reruns_sbatch_{exp_typ}_{hop}hop"
    os.makedirs(sbatch_dir, exist_ok=True)

    # Directory for SLURM stdout logs
    # Store in scratch on CC to avoid filling up home directory.
    log_dir = f"/home/rsaha/scratch/dm_alchemy"
    os.makedirs(log_dir, exist_ok=True)

    generated_files: List[str] = []

    for r in incomplete_runs:
        init_seed = r["init_seed"]
        frozen_layer = r["frozen_layer"]
        freeze_epoch = r["freeze_epoch"]
        checkpoint_path = r["checkpoint_base_path"]

        # Compute dynamic time allocation.
        if enable_chunked_resume:
            time_epochs = chunk_epochs
        else:
            time_epochs = TOTAL_EPOCHS - freeze_epoch
        time_limit = compute_slurm_time(time_epochs, sec_per_epoch)

        # Extract scheduler type from the checkpoint path
        scheduler_type = extract_scheduler_type(checkpoint_path)

        job_tag = f"{exp_typ}_{hop}hop_s{init_seed}_{frozen_layer}_e{freeze_epoch}"
        sbatch_filename = f"rerun_{job_tag}.sh"
        sbatch_filepath = os.path.join(sbatch_dir, sbatch_filename)

        # Build the accelerate launch command
        # Weight decay, and eta_min are dynamically loaded in train.py.
        train_cmd = (
            f"CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --gpu_ids=all --main_process_port=0 "
            f"src/models/train.py "
            f"--resume_from_checkpoint=True "
            f"--resume_checkpoint_epoch={freeze_epoch} "
            f"--resume_checkpoint_path={checkpoint_path} "
            f"--freeze_layers={frozen_layer} "
            f"--epochs=1000 "
            f"--enable_chunked_resume={'True' if enable_chunked_resume else 'False'} "
            f"--chunk_epochs={chunk_epochs} "
            f"--chunk_checkpoint_every={chunk_checkpoint_every} "
            f"--chunk_keep_last_n={chunk_keep_last_n} "
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
            f"--save_checkpoints=False "
            f"--store_predictions=True "
            f"--use_scheduler=True "
            f"--scheduler_type={scheduler_type} "
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

        chunk_state_dir = os.path.join(
            checkpoint_path,
            f"resume_from_epoch_{freeze_epoch}__freeze_{frozen_layer}",
            "chunk_state",
        )

        auto_chain_block = []
        if auto_chain_afterok:
            auto_chain_block = [
                "",
                "# Auto-chain next chunk after successful completion",
                f"CHUNK_STATE_DIR=\"{chunk_state_dir}\"",
                "STOP_MARKER=\"$CHUNK_STATE_DIR/autochain_stop.json\"",
                "TARGET_EPOCH=999",
                "LATEST_EPOCH=$(ls \"$CHUNK_STATE_DIR\"/chunk_checkpoint_epoch_*.pt 2>/dev/null | sed -E 's/.*chunk_checkpoint_epoch_([0-9]+)\\.pt/\\1/' | sort -n | tail -1)",
                "if [[ -f \"$STOP_MARKER\" ]]; then",
                "  echo \"No further auto-chaining needed (convergence marker found at $STOP_MARKER).\"",
                "elif [[ -n \"$LATEST_EPOCH\" && \"$LATEST_EPOCH\" -lt \"$TARGET_EPOCH\" ]]; then",
                "  echo \"Auto-chaining next chunk from epoch $LATEST_EPOCH (dependency afterok:$SLURM_JOB_ID).\"",
                "  sbatch --dependency=afterok:${SLURM_JOB_ID} \"$0\"",
                "else",
                "  echo \"No further auto-chaining needed (latest epoch: ${LATEST_EPOCH:-none}).\"",
                "fi",
            ]

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
            "set -euo pipefail",
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
            *auto_chain_block,
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
    p.add_argument("--manual_runs", type=str, default=None,
                    help="Path to a CSV file with explicit (init_seed,frozen_layer,freeze_epoch) "
                         "lines. Skips filesystem scanning and generates sbatch scripts "
                         "only for the listed runs.")
    p.add_argument("--export_manual_runs_csv", type=str, default=None,
                   help=("Export detected runs to a portable CSV suitable for --manual_runs. "
                         "In auto-scan mode exports incomplete runs; in manual mode exports loaded runs."))
    p.add_argument("--saved_models_root_override", type=str, default=None,
                   help=("Override cluster-based src/saved_models root when resolving checkpoint paths "
                         "for scan/manual CSV workflows."))
    p.add_argument("--complete_only", action="store_true",
                   help=("Print only completed runs and their prediction directories. "
                         "Suppresses scan summary and does not generate sbatch scripts."))
    p.add_argument("--max_missing_epochs", type=int, default=DEFAULT_MAX_MISSING_EPOCHS,
                    help=("Tolerance for missing prediction epochs before marking a run "
                          "as incomplete (default: 20)."))
    p.add_argument("--enable_chunked_resume", action="store_true",
                    help="Generate sbatch scripts that run chunked frozen resumes (requires train.py chunk flags).")
    p.add_argument("--chunk_epochs", type=int, default=200,
                    help="Max epochs to run per chunk when --enable_chunked_resume is used.")
    p.add_argument("--chunk_checkpoint_every", type=int, default=5,
                    help="Save chunk checkpoint every N epochs.")
    p.add_argument("--chunk_keep_last_n", type=int, default=5,
                    help="Number of chunk checkpoints to retain.")
    p.add_argument("--auto_chain_afterok", action="store_true",
                    help="If set, each generated sbatch script re-submits itself with --dependency=afterok when target epoch is not reached.")
    p.add_argument("--enable_metric_threshold_gate", action="store_true",
                  help=("If set, only status='incomplete' runs are additionally checked for "
                      "metric threshold crossing using stagewise pickles before generating reruns."))
    p.add_argument("--metric_threshold", type=float, default=DEFAULT_METRIC_THRESHOLD,
                  help=(f"Metric threshold for gate on {DEFAULT_METRIC_KEY} "
                      f"(default: {DEFAULT_METRIC_THRESHOLD})."))
    p.add_argument("--metric_patience", type=int, default=DEFAULT_METRIC_PATIENCE,
                  help=("Consecutive values required at/above threshold to mark a run as complete "
                      f"(default: {DEFAULT_METRIC_PATIENCE})."))
    return p.parse_args()


def main():
    args = parse_args()

    if args.output is None:
        args.output = f"resubmit_{args.exp_typ}_{args.hop}hop.sh"

    if args.metric_patience <= 0:
        print("ERROR: --metric_patience must be >= 1.")
        sys.exit(2)

    if args.manual_runs:
        if args.complete_only:
            print("ERROR: --complete_only is not supported with --manual_runs.")
            sys.exit(2)
        # ── Manual mode: load explicit run list, skip scanning ──
        runs = load_manual_runs(
            manual_runs_path=args.manual_runs,
            exp_typ=args.exp_typ,
            hop=args.hop,
            data_split_seed=args.data_split_seed,
            saved_models_root_override=args.saved_models_root_override,
        )
        print_report(runs)
        if args.export_manual_runs_csv:
            export_manual_runs_csv(runs, args.export_manual_runs_csv)
        if args.dry_run:
            print("\n[DRY RUN] Not writing sbatch scripts.")
        else:
            generate_sbatch_scripts(
                incomplete_runs=runs,
                output_path=args.output,
                exp_typ=args.exp_typ,
                hop=args.hop,
                account=args.account,
                enable_chunked_resume=args.enable_chunked_resume,
                chunk_epochs=args.chunk_epochs,
                chunk_checkpoint_every=args.chunk_checkpoint_every,
                chunk_keep_last_n=args.chunk_keep_last_n,
                auto_chain_afterok=args.auto_chain_afterok,
            )
    else:
        # ── Auto mode: scan filesystem for incomplete runs ──
        incomplete, complete_runs, threshold_met_runs, _metric_gate_diagnostics = find_incomplete_runs(
            exp_typ=args.exp_typ,
            hop=args.hop,
            data_split_seed=args.data_split_seed,
            init_seeds=args.init_seeds,
            max_missing_epochs=args.max_missing_epochs,
            saved_models_root_override=args.saved_models_root_override,
            enable_metric_threshold_gate=args.enable_metric_threshold_gate,
            metric_threshold=args.metric_threshold,
            metric_patience=args.metric_patience,
        )
        if args.export_manual_runs_csv:
            export_manual_runs_csv(incomplete, args.export_manual_runs_csv)
        if args.complete_only:
            print_complete_report(complete_runs)
        else:
            if args.enable_metric_threshold_gate:
                print_threshold_met_report(threshold_met_runs)
            print_report(incomplete)
            if args.dry_run:
                print("\n[DRY RUN] Not writing sbatch scripts.")
            else:
                generate_sbatch_scripts(
                    incomplete_runs=incomplete,
                    output_path=args.output,
                    exp_typ=args.exp_typ,
                    hop=args.hop,
                    account=args.account,
                    enable_chunked_resume=args.enable_chunked_resume,
                    chunk_epochs=args.chunk_epochs,
                    chunk_checkpoint_every=args.chunk_checkpoint_every,
                    chunk_keep_last_n=args.chunk_keep_last_n,
                    auto_chain_afterok=args.auto_chain_afterok,
                )


if __name__ == "__main__":
    main()
