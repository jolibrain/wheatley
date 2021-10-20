#!/bin/sh

python3 train.py --n_j 6 --n_m 6 --total_timesteps 1000000 --n_test_env 100 --eval_freq 5000 --n_steps_episode 256 --batch_size 128 --seed 1000 --cpu --n_workers 8 --add_force_insert_boolean 
