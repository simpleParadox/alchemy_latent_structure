# Cluster Execution Guide

This document details how to submit and execute training jobs on the Slurm cluster.

## Environment Setup

Before launching jobs, ensure that the dependencies are installed and `accelerate` is configured.

```bash
# Load CUDA and python modules as required by your cluster
module load cuda/11.8 python/3.10

# Activate your virtual environment
source venv/bin/activate
```

## Running with Accelerate

The codebase uses Hugging Face `accelerate` for distributed and multi-GPU training. The configurations for accelerate are stored in `configs/accelerate_config.yaml`.

You can launch training using:
```bash
accelerate launch --config_file configs/accelerate_config.yaml src/models/train.py [arguments...]
```

## Slurm Job Configuration

Job scripts are stored in the `slurm/` directory.

### Available Job Templates

- `slurm/job_tiny.sh`: Designed for debugging and fast runs on small models.
- `slurm/job_small.sh`: Configured for standard small-scale multi-GPU training.
- `slurm/job_large.sh`: Reserved for large models or heavy sweep runs.

### Submitting to the Queue

To submit a job script to the Slurm batch queue, use:
```bash
sbatch slurm/submit_job.sh
```

Ensure you modify the script resources (such as CPU, GPU, memory, and walltime) in the batch header comments before submission.
