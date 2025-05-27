CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch --main_process_port=0 train.py --batch_size=512 --wandb_mode='online' --epochs=100 --model_size='small' --num_workers=15 

CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch --main_process_port=0 train.py --batch_size=512 --wandb_mode='online' --epochs=100 --model_size='medium' --num_workers=15
