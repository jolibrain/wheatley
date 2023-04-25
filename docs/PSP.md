# PSP sample

Sample line for the famous '272' problem
```
python3 ./train_psp.py --load_problem instances/psp/272/272.sm --n_validation_env=1 --exp_name_appendix TEST_PSP_272 --n_steps_episode 1600 --n_workers 10 --batch_size 160 --n_epochs 20 --device cuda:0  --total_timesteps 100000
```


Sample line for the famous '272' problem with uncertainty
```
python3 ./train_psp.py --load_problem instances/psp/272/272.sm --n_validation_env=10 --exp_name_appendix TEST_PSP_272_UNC --n_steps_episode 1800 --n_workers 10 --batch_size 180 --n_epochs 20 --device cuda:0 --fixed_validation --total_timesteps 1000000 --generate_duration_bounds 0.05 0.2 --duration_type stochastic
```
