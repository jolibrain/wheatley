#!/bin/sh

python3 train.py --n_j 6 --n_m 6 --total_timesteps 2000000 --n_test_env 10 --eval_freq 5000 --n_steps_episode 128 --batch_size 64 --path="saved_networks/6j6m_2e6timesteps_0" --divide_loss --seed 0
python3 train.py --n_j 6 --n_m 6 --total_timesteps 2000000 --n_test_env 10 --eval_freq 5000 --n_steps_episode 128 --batch_size 64 --path="saved_networks/6j6m_2e6timesteps_1" --divide_loss --seed 1
python3 train.py --n_j 6 --n_m 6 --total_timesteps 2000000 --n_test_env 10 --eval_freq 5000 --n_steps_episode 128 --batch_size 64 --path="saved_networks/6j6m_2e6timesteps_2" --divide_loss --seed 2
python3 train.py --n_j 6 --n_m 6 --total_timesteps 2000000 --n_test_env 10 --eval_freq 5000 --n_steps_episode 128 --batch_size 64 --path="saved_networks/6j6m_2e6timesteps_3" --divide_loss --seed 3
python3 train.py --n_j 6 --n_m 6 --total_timesteps 2000000 --n_test_env 10 --eval_freq 5000 --n_steps_episode 128 --batch_size 64 --path="saved_networks/6j6m_2e6timesteps_4" --divide_loss --seed 4
python3 train.py --n_j 6 --n_m 6 --total_timesteps 2000000 --n_test_env 10 --eval_freq 5000 --n_steps_episode 128 --batch_size 64 --path="saved_networks/6j6m_2e6timesteps_5" --divide_loss --seed 5
python3 train.py --n_j 6 --n_m 6 --total_timesteps 2000000 --n_test_env 10 --eval_freq 5000 --n_steps_episode 128 --batch_size 64 --path="saved_networks/6j6m_2e6timesteps_6" --divide_loss --seed 6
python3 train.py --n_j 6 --n_m 6 --total_timesteps 2000000 --n_test_env 10 --eval_freq 5000 --n_steps_episode 128 --batch_size 64 --path="saved_networks/6j6m_2e6timesteps_7" --divide_loss --seed 7
python3 train.py --n_j 6 --n_m 6 --total_timesteps 2000000 --n_test_env 10 --eval_freq 5000 --n_steps_episode 128 --batch_size 64 --path="saved_networks/6j6m_2e6timesteps_8" --divide_loss --seed 8
python3 train.py --n_j 6 --n_m 6 --total_timesteps 2000000 --n_test_env 10 --eval_freq 5000 --n_steps_episode 128 --batch_size 64 --path="saved_networks/6j6m_2e6timesteps_9" --divide_loss --seed 9
python3 train.py --n_j 6 --n_m 6 --total_timesteps 2000000 --n_test_env 10 --eval_freq 5000 --n_steps_episode 128 --batch_size 64 --path="saved_networks/6j6m_2e6timesteps_10" --divide_loss --seed 10
python3 train.py --n_j 6 --n_m 6 --total_timesteps 2000000 --n_test_env 10 --eval_freq 5000 --n_steps_episode 128 --batch_size 64 --path="saved_networks/6j6m_2e6timesteps_11" --divide_loss --seed 11
