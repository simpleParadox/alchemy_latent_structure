# Evaluation Metrics and Stage Detection

This document explains the metrics and methodologies used to evaluate transformers on the latent structure of DM Alchemy.

## Key Metrics

### 1. Stone State Prediction Accuracy
- **Definition:** The percentage of query episodes where the model predicts the exact correct resulting stone state.
- **Support-Query Splits:** We measure accuracy separately on elements matching the support distribution vs. out-of-distribution elements.

### 2. Transition Dynamics (Stage Detection)
To detect when a model transitions between learning stages/phases:
- **Threshold ($\tau$):** We set a high threshold (typically $\tau = 0.95$ accuracy).
- **Transition Epoch:** The training epoch at which the running accuracy first exceeds $\tau$ and remains above it.
- **Delta-Time ($\Delta t$):** The number of epochs between learning adjacent stages (e.g., time elapsed between mastering $P(b|a)$ and $P(c|ab)$).

## Running Evaluation

Post-hoc evaluation is driven by scripts in `src/analysis/`:
- Use `src/analysis/analyze_predictions.py` to process validation files and write prediction accuracies into pickle format.
- Plot learning curves using `src/analysis/plot_overall_accuracy_curves_from_pickles.py` and `src/analysis/plot_stages_from_pickles.py`.
- Plots will be saved directly into the `all_images/` directory.
