# Experiment Log

Use this file to record and track hyperparameter sweeps, model checkpoints, and findings.

## Log Template

| Date | Run ID / W&B Link | Task/Hops | Model Config | LR / WD / Scheduler | Seed | Convergence Epoch | Val Acc (Support/Query) | Key Takeaways / Notes |
|------|-------------------|-----------|--------------|---------------------|------|-------------------|-------------------------|------------------------|
| YYYY-MM-DD | `run_1234abcd` | 2-hop comp | `decoder_xsmall` | 1e-4 / 0.001 / Cosine | 42 | 350 | 0.96 / 0.94 | Baseline convergence achieved on original reward chemistry. |

---

## Active Sweeps and Runs

*Document any active Weights & Biases sweeps here.*

- **Sweep ID:** (e.g. `user/project/sweep_abcd1234`)
- **Objective:** Hyperparameter optimization for learning rate and weight decay on 5-hop decomposition.
- **Config Path:** `configs/continual_wandb_sweep.yaml`
