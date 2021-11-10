#!/bin/sh

python3 train.py --n_j 6 --n_m 6 --total_timesteps 2000000 --n_test_env 100 --eval_freq 10000 --n_steps_episode 256 --batch_size 128 --seed 1000 --max_pool --ent_coef 0.001 --vf_coef 0.1 --cpu --n_workers 8 --add_force_insert_boolean --features duration total_job_time total_machine_time job_completion_percentage machine_completion_percentage mopnr --lr 0.00002 --custom_heuristic_name MOPNR --gconv_type gatv2 --exp_name_appendix 7_features_MOPNR_bis

python3 train.py --n_j 6 --n_m 6 --total_timesteps 2000000 --n_test_env 100 --eval_freq 10000 --n_steps_episode 256 --batch_size 128 --seed 1000 --max_pool --ent_coef 0.001 --vf_coef 0.1 --cpu --n_workers 8 --add_force_insert_boolean --features duration total_job_time total_machine_time job_completion_percentage machine_completion_percentage mwkr --lr 0.00002 --custom_heuristic_name MWKR --gconv_type gatv2 --exp_name_appendix 7_features_MWKR_bis

python3 train.py --n_j 6 --n_m 6 --total_timesteps 3000000 --n_test_env 100 --eval_freq 10000 --n_steps_episode 256 --batch_size 128 --seed 1000 --max_pool --ent_coef 0.001 --vf_coef 0.1 --cpu --n_workers 8 --add_force_insert_boolean --features duration total_job_time total_machine_time job_completion_percentage machine_completion_percentage mopnr mwkr --lr 0.00002 --gconv_type gatv2 --exp_name_appendix 7_features_MWKR_MOPNR
