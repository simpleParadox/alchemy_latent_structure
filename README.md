# Code for the paper:
Understanding the staged dynamics of of transformers in latent structure learning.

This codebase contains the code to train and evaluate transformer based models on the DM Alchemy dataset.

## Installation

This project is built as a python package (`dm-alchemy`) along with scripts for data processing and model training.

To install the environment and the required package dependencies, run:

```bash
pip install -e .
```

To run the model training scripts, you will also need the following deep learning libraries:

```bash
pip install torch accelerate wandb tqdm
```

## Usage

### Training Models

The main entry point for training is `src/models/train.py`. The script uses `argparse` to configure the dataset, model architecture, training loop, and optimizer.

**Example Command (Held-out Experiment):**

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
    --wandb_project <enter-your-wandb-project-name>
```

> **Note:** The `held_out` experiment type (exp_typ) is enabled by setting `--is_held_out_color_exp True`. For multi-GPU training, you can use `accelerate launch`.


### Weights & Biases (W&B) Sweeps

> **Note:** The `train.py` script is fully compatible with Weights & Biases sweeps.
> Because all hyperparameters are exposed via `argparse`, you can easily set up a `sweep.yaml` configuration to search over learning rates, model sizes, architectures, and scheduler configurations. The script will automatically pick up the arguments injected by the W&B agent.

### Control Flow Documentation

For a detailed breakdown of how the training pipeline operates and how different arguments affect the execution path (e.g., initialization, dataset building, and model selection), please refer to `control_flow.md`.
