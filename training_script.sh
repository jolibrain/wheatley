#!/bin/sh

for i in 0 1 2 3 4 5 6 7 8 9 
do
    python3 train.py --n_j 6 --n_m 6 --total_timesteps 1000000 --n_test_env 100 --eval_freq 5000 --n_steps_episode 512 --batch_size 256 --seed $i
done
