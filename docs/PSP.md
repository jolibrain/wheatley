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

### Inference

Once the model is trained, you can use it to solve new problems using:

```sh
TODO
```
