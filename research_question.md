# Research Questions: Staged Dynamics in Latent Structure Learning

This document outlines the core scientific questions under investigation in this project.

## Overview
We study how transformer-based models learn the latent causal/logical structure of the **DM Alchemy** environment. Specifically, we investigate the **staged/phasic learning dynamics** of these models—understanding whether and why they learn specific topological properties in discrete stages.

## Core Questions

### 1. Phasic Learning Dynamics
- **Question:** Does the learning of latent structure proceed via discrete phases/stages or continuous gradients?
- **Hypothesis:** Models learn basic relations (e.g. direct edges or 2-hop compositions) first, which serve as building blocks for learning higher-hop compositions and decomposition rules later in training.
- **Metrics:** We track the transition times ($t$) between learning stages using a threshold accuracy (typically $\tau = 0.95$).

### 2. Compositionality vs. Decomposition
- **Question:** How do transformers differ when learning compositional paths versus decomposing complex interactions?
- **Investigation:** We compare performance profiles and epoch-wise learning rates between *compositional sweeps* (e.g., 2-hop to 5-hop) and *decomposition tasks*.

### 3. Out-of-Distribution (OOD) Generalization
- **Question:** Does staged learning enable the model to generalize compositionally to held-out paths?
- **Investigation:** We test models on held-out chemistry graph splits (e.g., 4-edge held-out experiments) to assess whether they learn the true underlying group structure or rely on localized shortcuts.

### 4. Circuit and Representation Analysis
- **Question:** What internal mechanisms or attention heads mediate these transitions?
- **Investigation:** Using mechanical interpretability techniques (such as activation patching and caching), we map the emergence of specialized heads (e.g., stone identity tracking, potion mapping) over the course of training epochs.
