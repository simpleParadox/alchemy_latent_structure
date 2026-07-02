# Instructions for AI Coding Agents

Welcome! If you are an AI coding agent pair programming with a user on this codebase, please review and follow these guidelines to maintain repository integrity and style.

## Repository Context

This repository contains code for training and evaluating transformer-based models on the **DM Alchemy** dataset, specifically studying how models learn latent structures in a staged/phasic manner.

## Key Directory Structure

- `src/models/`: Contains the primary training pipeline (`train.py`, `train_continual.py`), network architectures (`models.py`), and dataset loaders (`data_loaders.py`).
- `src/mech_interp/`: Holds mechanical interpretability scripts such as activation patching and caching.
- `src/analysis/`: Contains all analysis, comparison, and plotting utilities.
- `configs/`: Directory for Weights & Biases configuration sweeps and YAML settings.
- `slurm/`: Job submission shell scripts and batch templates.
- `all_images/`: Consolidates all generated plots, PNGs, and PDF figures.
- `results/`: Local output files (checkpoints, metrics, etc.), which are gitignored.

## Coding Guidelines

### 1. Maintain Documentation Integrity
Do not delete or modify existing documentation unless explicitly requested. Always keep `control_flow.md` and code comments up to date.

### 2. Follow established patterns
- Use absolute imports under the `src` directory (e.g., `from src.models.data_loaders import AlchemyDataset`).
- When adding or modifying command-line options, ensure they are integrated with `argparse` in `train.py` and documented in `control_flow.md`.
- Handle device allocation via Hugging Face `accelerate` (e.g., using `Accelerator` from `accelerate` package) rather than manual `.cuda()` calls.

### 3. File Operations
- **Never delete any files** unless directly commanded by the user.
- Place all temporary run scripts or draft diagnostics in `results/` or in the system's scratch directory.
