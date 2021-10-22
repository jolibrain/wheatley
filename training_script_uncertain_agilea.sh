for i in 0 1 2 3 4 5 6 7 8 9
do
    python3 train.py --n_j 12 --n_m 16 --total_timesteps 1000000 --n_test_env 10 --eval_freq 500 --n_steps_episode 512 --batch_size 256 --seed $i --fixed_distrib --reward_model_config Sparse --features duration  --exp_name_append REBASED   --lr 0.0001 --gconv_type gatv2 --ortools_strategy averagistic --load_problem instances/agilea/small_12_unc.txt
done
