# Solving RCPSP Problems

Wheatley can train a RCPSP solver for determinists and stochastics instances.
Once trained, the solver can be used to solve new and bigger instances.

## Quickstart

To launch a training you first have to install the dependencies (see README).
Once that's done, you can launch the visdom logging server:

```sh
python -m visdom.server
```

Trainings are displayed on [localhost](http://localhost:8097).
More information about visdom [here](https://github.com/fossasia/visdom).

### Training

Launch a training run:

```sh
python3 -m psp.train_psp \
	--batch_size 50 \
	--clip_range 0.25 \
	--conflicts att \
	--device cuda:0 \
	--exp_name_appendix rcpsp_test \
	--fixed_validation \
	--gae_lambda 0.95 \
	--graph_pooling max \
	--hidden_dim_actor 32 \
	--hidden_dim_critic 32 \
	--hidden_dim_features_extractor 64 \
	--keep_past_prec \
	--layer_pooling last \
	--load_problem ./instances/psp/272/272.sm \
	--lr 1.0e-5 \
	--n_epochs 3 \
	--n_layers_features_extractor 6 \
	--n_mlp_layers_actor 1 \
	--n_mlp_layers_critic 1 \
	--n_mlp_layers_features_extractor 1 \
	--n_steps_episode 13000 \
	--n_workers 5 \
	--residual_gnn \
	--target_kl 0.2 \
	--total_timesteps 10000000 \
	--use_old_resource_info \
	--vecenv_type graphgym \
	--weight_decay 0.00
```

This will launch a training for the 'famous' 272 problem.
You can choose the instance to train on by using `--load-problem` argument.


## Reproducing JSSP paper results with PSP code:

### Deterministic results
```
python3  -m psp.train_psp \
        --batch_size 256 \
        --conflicts node \
        --device cuda:0 \
        --exp_name_appendix rcpsp_jssppaper_det \
        --fixed_validation \
        --gae_lambda 1.0 \
        --graph_pooling learn \
        --hidden_dim_features_extractor 64 \
        --n_epochs 3 \
        --n_layers_features_extractor 8 \
        --n_steps_episode 9500 \
        --n_workers 10 \
        --path /tmp/saved_networks/ \
        --test_dir ./instances/psp/taillards/6x6/ \
        --total_timesteps 100000000 \
        --n_j 6 \
        --n_m 6 \
        --random_taillard \
        --duration_mode_bounds 1 100 \
        --residual_gnn \ 
        --vecenv_type graphgym 
```

### Stochastic results
```
python3 -m psp.train_psp \
        --batch_size 256 \
        --conflicts clique \
        --device cuda:0\
        --exp_name_appendix rcpsp_jssppaper_stoch \
        --fixed_validation \
        --gae_lambda 1.0 \
        --graph_pooling learn \
        --hidden_dim_features_extractor 64 \
        --n_epochs 3 \
        --n_layers_features_extractor 8 \
        --n_steps_episode 9500 \
        --n_workers 10 \
        --path /tmp/saved_networks/ \
        --total_timesteps 100000000 \
        --n_j 6 \
        --n_m 6 \
        --random_taillard \
        --duration_mode_bounds 10 50 \
        --residual_gnn \
        --duration_type stochastic \
        --duration_delta 10 200 \
        --vecenv_type graphgym \
        --n_validation_env 100 \
        --ortools_strategy realistic optimistic pessimistic averagistic
```
