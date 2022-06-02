time python3 -m cProfile -o profile.data train.py --vecenv_type dummy --n_j 32 --n_m 32 --total_timesteps 1024 --validation_freq 1025 --exp_name_appendix profile --device cuda:0
