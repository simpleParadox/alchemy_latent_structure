"""
wandb_stage_filters.py

W&B query filters for the survivorship bias stage analysis.
Each entry maps (exp_type, hop) to a MongoDB-style filter dict
that precisely identifies that experimental condition in the
simpleparadox/alchemy-meta-learning project.

preprocessed_dir is the most reliable discriminator because it
is set from the sweep config *before* wandb.init() is called,
so the original relative path value is what gets stored in the run config.
"""

import datetime

# Only consider runs created after this date
CUTOFF_DATE = datetime.datetime(2025, 8, 1, tzinfo=datetime.timezone.utc)
CUTOFF_STR  = CUTOFF_DATE.strftime("%Y-%m-%dT%H:%M:%S")

# Preprocessed directory substrings that identify each experiment type
_PREPDIR = {
    "held_out":      "src/data/shuffled_held_out_exps_preprocessed_separate_enhanced",
    "composition":   "src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_preprocessed",
    "decomposition": "src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes",
    # "decomposition": "src/data/decomposition_length_matched_subsampled_grouped_by_end_state_preprocessed"
}

# ---------------------------------------------------------------------------
# Hyperparameter constraints (applied to ALL tasks)
# ---------------------------------------------------------------------------
_WEIGHT_DECAY_VALUES = [0.1, 0.01, 0.001]
_ETA_MIN_VALUES      = [7e-5, 8e-5, 9e-5, 8.5e-5, 9.5e-5, 1e-5]
# _SCHEDULER_VALUES    = ["cosine", "cosine_restarts"]
_SCHEDULER_VALUES    = ["cosine"]
_LR_VALUES           = [0.0001]

_COMMON_HP_FILTERS = {
    "config.weight_decay":    {"$in": _WEIGHT_DECAY_VALUES},
    "config.eta_min":         {"$in": _ETA_MIN_VALUES},
    "config.scheduler_type":  {"$in": _SCHEDULER_VALUES},
    "config.learning_rate":   {"$in": _LR_VALUES},
    "config.model_size":      "xsmall",
    "config.data_split_seed": {"$in": [0]},
    # Exclude resumed runs (they are continuations, not independent runs)
    "config.resume_from_checkpoint": {"$nin": [True, "True", "true"]},
}

# ---------------------------------------------------------------------------
# Expected seed sets per (exp_type, hop)
# (decomposition 5-hop uses seed=2 instead of seed=1)
# ---------------------------------------------------------------------------
EXPECTED_SEEDS: dict[tuple, list] = {
    ("held_out",      1): [42, 1, 3],
    ("composition",   2): [42, 1, 3],
    ("composition",   3): [42, 1, 3],
    ("composition",   4): [42, 1, 3],
    ("composition",   5): [42, 1, 3],
    ("decomposition", 2): [42, 1, 3],
    ("decomposition", 3): [42, 1, 3],
    ("decomposition", 4): [42, 1, 3],
    ("decomposition", 5): [42, 2, 3],   # seed=2 instead of seed=1
}

# Expose HP lists publicly so the main script can build expected combo sets
EXPECTED_WEIGHT_DECAYS  = _WEIGHT_DECAY_VALUES
EXPECTED_ETA_MINS       = _ETA_MIN_VALUES
EXPECTED_SCHEDULERS     = _SCHEDULER_VALUES
EXPECTED_LEARNING_RATES = _LR_VALUES
EXPECTED_DATA_SEEDS     = [0]

# Build per-(exp_type, hop) W&B API filter dicts.
# Composition hop-N:   shop_length=1, qhop_length=N
# Decomposition hop-N: shop_length=N, qhop_length=1
# Held-out:            shop_length=1, qhop_length=1  (distinct preprocessed_dir)

WANDB_FILTERS: dict[tuple, dict] = {}

# --- Held-out (single hop value, seeds 1, 3, 42) ---
WANDB_FILTERS[("held_out", 1)] = {
    "created_at":              {"$gte": CUTOFF_STR},
    "config.preprocessed_dir": {"$in": [_PREPDIR["held_out"]]},
    "config.seed":             {"$in": [42, 1, 3]},
    **_COMMON_HP_FILTERS,
}

# --- Composition (2-hop through 5-hop, seeds 1, 3, 42) ---
for _hop in [2, 3, 4, 5]:
    WANDB_FILTERS[("composition", _hop)] = {
        "created_at":              {"$gt": CUTOFF_STR},
        "config.preprocessed_dir": {"$regex": _PREPDIR["composition"]},
        "config.shop_length":      1,
        "config.qhop_length":      _hop,
        "config.seed":             {"$in": [42, 1, 3]},
        **_COMMON_HP_FILTERS,
    }

# --- Decomposition (2-hop through 4-hop, seeds 1, 3, 42) ---
for _hop in [2, 3, 4]:
    WANDB_FILTERS[("decomposition", _hop)] = {
        "created_at":              {"$gte": CUTOFF_STR},
        "config.preprocessed_dir": {"$regex": _PREPDIR["decomposition"]},
        "config.shop_length":      _hop,
        "config.qhop_length":      1,
        "config.seed":             {"$in": [42, 1, 3]},
        **_COMMON_HP_FILTERS,
    }

# --- Decomposition 5-hop (different seed set: 42, 2, 3) ---
WANDB_FILTERS[("decomposition", 5)] = {
    "created_at":              {"$gte": CUTOFF_STR},
    "config.preprocessed_dir": {"$regex": _PREPDIR["decomposition"]},
    "config.shop_length":      5,
    "config.qhop_length":      1,
    "config.seed":             {"$in": [42, 2, 3]},
    **_COMMON_HP_FILTERS,
}


# ---------------------------------------------------------------------------
# Stage thresholds (val_epoch_accuracy as a fraction in [0, 1])
# ---------------------------------------------------------------------------
STAGE1_LO = 0.10   # Lower bound for "P[A] learned" band (~12.5 %)
STAGE1_HI = 0.16   # Upper bound for "still only P[A]" band

STAGE3_THRESHOLD = 0.95   # P[C|A,B] fully learned

# Stage 2 threshold per (exp_type, hop)
STAGE2_THRESHOLD: dict[tuple, float] = {
    ("held_out",      1): 0.36,   # ~38 %  – P[C|A,B] for held-out
    ("composition",   2): 0.30,   # ~33 %  – P[B|A] for 2-hop
    ("composition",   3): 0.22,   # ~25 %  – P[B|A] for 3-hop
    ("composition",   4): 0.30,   # ~33 %  – P[B|A] for 4-hop
    ("composition",   5): 0.22,   # ~25 %  – P[B|A] for 5-hop
    ("decomposition", 2): 0.22,   # ~25 %  – P[B|A] for all decomp hops
    ("decomposition", 3): 0.22,
    ("decomposition", 4): 0.22,
    ("decomposition", 5): 0.22,
}

# Epoch budgets
EPOCH_BUDGET: dict[str, int] = {
    "held_out":      1000,
    "composition":   1000,
    "decomposition": 1000,
}

# Number of consecutive logged points that must satisfy the threshold
SUSTAINED_WINDOW = 3
