#!/bin/bash

# Simple multi-GPU training script using Hugging Face Accelerate
# Make sure you have accelerate installed: pip install accelerate

# Activate environment
source alchemy_env/bin/activate

# Method 1: Use accelerate launch with automatic configuration
echo "Launching training with accelerate..."
python -m accelerate.commands.launch src/models/train.py \
    --task_type classification \
    --train_data_path src/data/generated_data/compositional_chemistry_samples_167424_train_shop_1_qhop_1.json \
    --val_data_path src/data/generated_data/compositional_chemistry_samples_167424_val_shop_1_qhop_1.json \
    --model_size small \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_workers 4 \
    --log_interval 10 \
    --wandb_mode offline

# Method 2: Use accelerate launch with custom config (uncomment to use)
# python -m accelerate.commands.launch --config_file accelerate_config.yaml src/models/train.py \
#     --task_type classification \
#     --train_data_path src/data/generated_data/compositional_chemistry_samples_167424_train_shop_1_qhop_1.json \
#     --val_data_path src/data/generated_data/compositional_chemistry_samples_167424_val_shop_1_qhop_1.json \
#     --model_size small \
#     --epochs 2 \
#     --batch_size 16 \
#     --learning_rate 1e-4 \
#     --num_workers 4 \
#     --log_interval 10 \
#     --wandb_mode offline

# Method 3: For multi-GPU training, specify number of processes
# python -m accelerate.commands.launch --multi_gpu --num_processes=2 src/models/train.py \
#     --task_type classification \
#     --train_data_path src/data/generated_data/compositional_chemistry_samples_167424_train_shop_1_qhop_1.json \
#     --val_data_path src/data/generated_data/compositional_chemistry_samples_167424_val_shop_1_qhop_1.json \
#     --model_size small \
#     --epochs 2 \
#     --batch_size 16 \
#     --learning_rate 1e-4 \
#     --num_workers 4 \
#     --log_interval 10 \
#     --wandb_mode offline
