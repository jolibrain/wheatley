# Examples

To get live metrics, start a Visdom server on port 8097:
```
python3 -m visdom.server
```

## Training over a fixed random problem

```
python3 train.py --n_j 6 --n_m 6 --total_timesteps 1000000 --n_validation_env 1 --n_steps_episode 360 --batch_size 360 --seed 1 --gconv_type gatv2 --fixed_problem --lr 0.0002
```

## Training over a custom fixed problem

```
python3 train.py --n_j 12 --n_m 16 --total_timesteps 1000000 --n_validation_env 1 --n_steps_episode 192 --batch_size 192 --seed 2 --lr 0.0002 --gconv_type gatv2 --fixed_problem --load_problem instances/agilea/small_12.txt --n_epochs 3
```

## Training over a standard Taillard problem

```
cd benchmark/
python3 run_taillard.py --lr 0.0002 --gconv_type gatv2 --total_timesteps 5000000 --taillard_pb ta01 --n_epochs 3
```

## Training over a set of fixed-size random problems

```
python3 train.py --n_j 6 --n_m 6 --total_timesteps 1000000 --n_validation_env 1 --n_steps_episode 360 --batch_size 360 --seed 1 --gconv_type gatv2 --lr 0.0002
```

## Training over a fixed random problem with uncertain durations

```
python3 train.py --n_j 6 --n_m 6 --total_timesteps 1000000 --n_validation_env 1 -n_steps_episode 360 --batch_size 360 --seed 1 --gconv_type gatv2 --lr 0.0002 --duration_type stochastic --fixed_problem --features duration --reward_model_config optimistic --ortools_strategy averagistic
```

## Training over a custom problem with uncertain durations

```
python3 train.py --n_j 12 --n_m 16 --total_timesteps 4000000 --n_validation_env 10 --n_steps_episode 1920 --batch_size 480 --duration_type stochastic --fixed_problem --features duration --exp_name_appendix uncertainty --lr 0.00002 --load_problem instances/agilea/small_12_unc.txt --exp_name_appendix test_agilea_uncertain3 --n_epochs 3 --reward_model_config optimistic --ortools_strategy averagistic
```
