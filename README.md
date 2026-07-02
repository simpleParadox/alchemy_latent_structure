# Understanding the Staged Dynamics of Transformers in Latent Structure Learning

This repository contains the codebase to train and evaluate transformer-based models on the DM Alchemy dataset, investigating the staged/phasic dynamics of how models learn latent graphical chemistry structures.

## Directory Structure

The repository is organized as follows:

```
alchemy_latent_structure/
├── AGENTS.md                  ← AI coding agent instructions and rules
├── README.md                  ← Installation and usage guide
├── research_question.md       ← Scientific goals and questions under study
├── experiment_log.md          ← Log format for tracking runs and hyperparameter sweeps
├── cluster.md                 ← Instruction guide for cluster/Slurm execution
├── dataset.md                 ← Dataset topology details (hops, composition vs decomposition)
├── evaluation.md              ← Metrics, thresholding (tau=0.95), and stage detection
├── setup.py                   ← Package installation script
├── slurm/                     ← Consolidates all Slurm job templates and launch scripts
│   ├── job_small.sh
│   ├── job_large.sh
│   ├── job_tiny.sh
│   └── ...
├── configs/                   ← Consolidates Weights & Biases configs, accelerate yaml config
│   ├── accelerate_config.yaml
│   ├── continual_wandb_sweep.yaml
│   └── ...
├── src/
│   ├── data/                  ← Shuffled json datasets and support-query generators
│   ├── models/                ← Core training and validation scripts
│   │   ├── train.py
│   │   ├── train_continual.py
│   │   ├── models.py
│   │   ├── data_loaders.py
│   │   └── val.py
│   ├── mech_interp/           ← Mechanical interpretability scripts (e.g. activation caching)
│   ├── analysis/              ← Post-hoc prediction analyzers and plotting scripts
│   │   ├── analyze_predictions.py
│   │   ├── plot_stages_from_pickles.py
│   │   └── ...
│   ├── utils/                 ← Utilities and support code
│   └── notebooks/             ← Jupyter notebooks for interactive analysis
├── all_images/                ← Consolidates all output figures, plots (PNG, PDF)
├── json_files/                ← Consolidates all root-level JSON metrics and watchdog logs
├── csv_files/                 ← Consolidates all root-level CSV tracking logs and metrics
├── pkl_files/                 ← Consolidates all root-level PKL dataset cache/head sweep results
├── txt_files/                 ← Consolidates all root-level TXT command logs, logs, and token configs
└── results/                   ← Directory for local output files (gitignored)
```

## Installation

This project is built as a python package (`dm-alchemy`). To install the package in editable mode along with its dependencies:

```bash
pip install -e .
```

To run training, you will also need:

```bash
pip install torch accelerate wandb tqdm
```

## Usage

### Training Models
The main entry point for training is `src/models/train.py`.

```bash
python src/models/train.py \
    --task_type classification \
    --model_architecture decoder \
    --model_size xsmall \
    --is_held_out_color_exp True \
    --train_data_path src/data/shuffled_held_out_exps_generated_data_enhanced/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp.json \
    --val_data_path src/data/shuffled_held_out_exps_generated_data_enhanced/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp.json \
    --seed 42 \
    --epochs 1000 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --weight_decay 0.001 \
    --eta_min 7e-5 \
    --wandb_project <your-project>
```

For multi-GPU execution using the cluster config, run:
```bash
accelerate launch --config_file configs/accelerate_config.yaml src/models/train.py [args...]
```

### Analysis & Evaluation
After a run completes, you can analyze validation predictions:
```bash
python src/analysis/analyze_predictions.py --model_dir results/checkpoints/ --val_data_path src/data/...
```

To plot the learning stages from pickles:
```bash
python src/analysis/plot_stages_from_pickles.py --pickle_dir results/pickle_files/ --output_dir all_images/
```

## Documentation

For deep dives into specific topics, see:
- [AGENTS.md](file:///home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/AGENTS.md): AI guidelines
- [research_question.md](file:///home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/research_question.md): Core scientific goals
- [experiment_log.md](file:///home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/experiment_log.md): Track runs
- [cluster.md](file:///home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/cluster.md): Slurm execution guide
- [dataset.md](file:///home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/dataset.md): Alchemy graph dataset structure
- [evaluation.md](file:///home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/evaluation.md): Transition and threshold metrics
- [control_flow.md](file:///home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/control_flow.md): Program flow documentation
