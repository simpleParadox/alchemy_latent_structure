# DM Alchemy Dataset Structure

This document describes the structure of the generated datasets used for learning latent representations of potion-stone interactions.

## Environment Overview

In **DM Alchemy**, stones have visual features (size, color, pattern) that map to a latent chemistry graph. Potion applications correspond to transitions on this graph. 

## Dataset Formats

The datasets are stored as JSON files under `src/data/` (or generated dynamically using `alchemy_prompt_generator.py`). Each sample represents an in-context learning task containing:
1. **Support Set:** A sequence of demonstrative episodes showing potion effects on various stones.
2. **Query Set:** A test episode where the model must predict the outcome of a potion application based on the support set.

## Tasks and Hops

Our experiments classify tasks by **Hops** and **Task Types**:

### 1. Latent Graph Hops
- **2-Hop:** Relations representing two sequential potion effects (paths of length 2 in the graph).
- **3-Hop, 4-Hop, 5-Hop:** Progressively longer paths in the graph, representing highly complex multi-step transitions.

### 2. Composition vs. Decomposition
- **Composition Tasks:** Learning to predict the final stone state after sequentially applying multiple potions ($A \xrightarrow{p_1} B \xrightarrow{p_2} C$).
- **Decomposition Tasks:** Given the start and end stone states, predicting the intermediate potion/states that occurred.

## Shuffled and Preprocessed Data

- Raw JSON datasets are shuffled and partitioned into training and validation splits.
- Preprocessed datasets are compiled into Python pickle files (`_data.pkl`, `_vocab.pkl`) under `src/data/` subdirectories to optimize dataloader read performance.
