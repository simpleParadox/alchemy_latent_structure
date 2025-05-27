CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py --batch_size=512 --wandb_mode='online' --epochs=100 --model_size='tiny' --num_workers=15

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py --batch_size=512 --wandb_mode='online' --epochs=100 --model_size='xsmall' --num_workers=15