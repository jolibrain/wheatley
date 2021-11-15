for i in 0 1 2 3 4 5 6 7 8 9
do
    python3 train.py --n_j 6 --n_m 6 --total_timesteps 1000000 --n_validation_env 10 --validation_freq 500 --n_steps_episode 512 --batch_size 256 --seed $i --duration_type stochastic --fixed_problem --reward_model_config Sparse --features duration  --exp_name_append REBASED   --lr 0.0001 --gconv_type gatv2 --ortools_strategy optimistic
done
