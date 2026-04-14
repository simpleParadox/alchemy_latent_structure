# Control Flow in `train.py`

This document details the control flow across the different Python files for training the models based on `train.py` and its interacting modules (`data_loaders.py` and `models.py`). It specifically highlights how `argparse` arguments alter the flow of the program.

## 1. Initialization and Setup (`train.py`)

- **Argument Parsing**: Execution begins in `main()`, where `parse_args()` is called to retrieve the hyperparameters and configurations for the run.
- **Resuming Checkpoints**:
    - If `--resume_from_checkpoint=True`, the code loads a checkpoint specified by `--resume_checkpoint_path` and `--resume_checkpoint_epoch`.
    - `validate_resume_compatibility()` verifies that crucial arguments (like `task_type`, `model_size`, `model_architecture`) match the ones in the checkpoint.
- **Environment and Paths**:
    - Based on the imported `cluster` variable from `cluster_profile` (and modified dynamically), the script establishes the `base_path` for the file system.
    - Path logic adjusts `train_data_path`, `val_data_path`, and `save_dir` to absolute paths relative to `base_path`.
    - `--data_split_seed` is dynamically inserted into the data filenames.
- **Accelerator Setup**:
    - `Accelerator` from Hugging Face is initialized. If `--fp16=True`, mixed precision is enabled.
    - `set_seed(--seed)` ensures deterministic execution.
- **Logging**:
    - Weights & Biases (WandB) is initialized on the main process with the parsed arguments, taking into account `--wandb_mode`, `--wandb_project`, `--wandb_entity`, and whether it's resuming an existing run (`--resume_wandb_run_id`).

## 2. Dataset and DataLoader Creation (`data_loaders.py`)

The main dataset class, `AlchemyDataset`, is instantiated to prepare the training data. The data loading flow diverges heavily based on the `--task_type` and input/output formats.

- **Initialization of `AlchemyDataset`**:
    - Depending on `--task_type` (`seq2seq`, `classification`, `classification_multi_label`, `seq2seq_stone_state`), the dataset determines default `--input_format` and `--output_format`. For example, `classification` defaults to `stone_states` for both, while `seq2seq` uses `features`.
    - It searches for preprocessed data in `--preprocessed_dir` if `--use_preprocessed=True`. If not found, it preprocesses raw JSON data from scratch.
- **Vocabulary Building**:
    - Separate input and output vocabularies are built based on the `--input_format` and `--output_format`.
    - Example: `_build_stone_state_potion_vocab()` builds vocabulary with whole stone states as single tokens, whereas `_build_feature_potion_vocab()` builds it using individual features from the stones.
- **Data Processing (`process_episode_worker`)**:
    - **`seq2seq`**: Both encoder and decoder inputs/targets are constructed using individual features as tokens.
    - **`classification`**: Encoder inputs consist of stone states or features. The target is a single class ID for the output stone state.
    - **`classification_multi_label`**: The target is a multi-hot feature vector. An `autoregressive` target vector is also created if needed.
    - **`seq2seq_stone_state`**: Uses complete stone states as individual tokens for the seq2seq format.
    - If `--filter_query_from_support=True`, queries are removed from the support set.
- **Validation Split**:
    - If `--val_split` is provided, `_create_train_val_splits()` partitions the dataset. For `classification`, it performs stratified splitting to balance classes. For others, it's a random split.
    - Otherwise, a separate validation file (`--val_data_path`) is loaded into a new `AlchemyDataset`.
- **DataLoaders**:
    - `DataLoader` is created using a custom `collate_fn`.
    - `collate_fn` pads sequences using `--padding_side`. For `model_architecture='decoder'`, it handles left or right padding appropriately.
    - Truncation is applied if `--use_truncation=True` and sequences exceed `--max_seq_len`.

## 3. Model Creation (`models.py`)

The model instantiated depends on `--task_type` and `--model_architecture`:

- **`seq2seq` and `seq2seq_stone_state`**:
    - Calls `create_transformer_model()` to instantiate a `Seq2SeqTransformer`. This model has an encoder, a decoder, and uses standard sequence-to-sequence loss (CrossEntropy).
- **`classification`**:
    - **`model_architecture="encoder"`**: Calls `create_classifier_model()` to instantiate a `StoneStateClassifier`. Uses the specified `--pooling_strategy` (global, query_only, last_token) to aggregate the encoder sequence into a single feature vector before classification.
    - **`model_architecture="decoder"`**: Calls `create_decoder_classifier_model()` to instantiate a `StoneStateDecoderClassifier`. This uses a causal mask and uses the last valid token's representation for classification. It can also utilize `--use_flash_attention=True`.
    - **`model_architecture="linear"`**: Calls `create_linear_model()` to instantiate a `LinearBaseline`. If `--flatten_linear_model_input=True`, the entire sequence is flattened instead of pooled.
- **`classification_multi_label`**:
    - Similar branching to `classification` (between encoder and decoder architectures). The model's output head size corresponds to the number of output features, representing a group-level multi-class setup.
- **Layer Freezing**:
    - If `--freeze_layers` is provided, `_apply_freeze_layers_in_place()` iterates over the model and sets `requires_grad=False` for the specified layers.

## 4. Optimizer and Scheduler

- **Optimizer**:
    - Instantiated based on `--optimizer` (`adamw`, `adam`, `rmsprop`, `adagrad`). Only parameters with `requires_grad=True` are passed to the optimizer.
- **Scheduler**:
    - If `--use_scheduler=True`, a learning rate scheduler is set up based on `--scheduler_type` (`cosine`, `cosine_restarts`, `exponential`, `reduce_on_plateau`, `sequential_lr`, `step_lr`, `multi_step_lr`).
    - The step timing (`--scheduler_call_location`) dictates whether the scheduler is updated `after_batch` or `after_epoch`.

## 5. Training and Validation Loop (`train.py`)

The execution iterates through the specified `--epochs`:

- **Training (`train_epoch`)**:
    - **`seq2seq` (Autoregressive vs. Teacher Forcing)**:
        - If `--use_autoregressive_training=True`, the model uses `model.generate()` to predict the sequence token by token. Loss is calculated based on token error rate.
        - Otherwise, teacher forcing is used, calculating loss on standard logits via CrossEntropy.
    - **`classification`**: Calculates CrossEntropy loss between the logits from the pooling/last token and the target class ID.
    - **`classification_multi_label`**: Iterates through predefined feature groups (e.g., color, size, shape) and sums the CrossEntropy loss for each group.
    - Gradients are clipped to `1.0`. `optimizer.step()` is called. `scheduler.step()` is invoked if configured for `after_batch`.
- **Validation (`validate_epoch`)**:
    - Operates similarly to `train_epoch` but without gradient calculations (`torch.no_grad()`).
    - **`seq2seq`**: Always uses `model.generate()` (autoregressive) to calculate true generation accuracy.
    - **Predictions Storage**: If `--store_predictions=True`, the predictions, targets, and inputs are accumulated and subsequently saved to disk using `np.savez_compressed`.
    - Handles model ablations (`ablation_type`) in `validate_epoch` to replace or zero out rewards embeddings or attention, typically used during custom evaluation loops outside of regular training.
- **Model Checkpointing**:
    - If the validation loss improves (`val_loss < best_val_loss`), the model state, optimizer state, scheduler state, and vocabularies are saved.
    - The saving path incorporates the rich hierarchical folder structure detailing the experiment's configurations (e.g., model size, architecture, task type, hop lengths).
    - If `--store_in_scratch=True`, the checkpoints are rerouted to a scratch directory for performance or quota reasons.
