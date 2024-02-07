# Solving JSSP Problems

## Quickstart

### Visdom

Launch the visdom logging server:

```sh
python -m visdom.server
```

Trainings are displayed on [localhost](http://localhost:8097).
More information about visdom [here](https://github.com/fossasia/visdom).

### Training

Launch a training run:

```sh
python3 -m jssp.train\
    --batch_size 245\
    --clip_range 0.20\
    --custom_heuristic_names SPT MWKR MOPNR FDD/MWKR\
    --device cuda:0\
    --duration_type deterministic\
    --ent_coef 0.05\
    --exp_name_appendix QUICKSTART_RUN\
    --fe_type dgl\
    --fixed_validation\
    --gae_lambda 0.99\
    --gamma 1.00\
    --graph_has_relu\
    --graph_pooling max\
    --hidden_dim_actor 32\
    --hidden_dim_critic 32\
    --hidden_dim_features_extractor 64\
    --layer_pooling last\
    --lr 1e-4\
    --max_n_j 100\
    --max_n_m 30\
    --mlp_act gelu\
    --n_epochs 3\
    --n_j 10\
    --n_layers_features_extractor 10\
    --n_m 10\
    --n_mlp_layers_actor 1\
    --n_mlp_layers_critic 1\
    --n_mlp_layers_features_extractor 1\
    --n_steps_episode 9800\
    --n_validation_env 100\
    --n_workers 1\
    --optimizer adamw\
    --ortools_strategy realistic\
    --residual_gnn\
    --seed 0\
    --target_kl 0.04\
    --total_timesteps 10_000_000\
    --validation_freq 3\
    --vf_coef 2.0
```

The important parameters here are `--n_j` and `--n_m`, to specify the number of
jobs and machines of the JSSP instances the model will be trained on.
`--max_n_j` and `--max_n_m` is used to define the starting feature space of the model.
Once trained, the model can be used to solve instances of maximal sizes
$maxn_j \times maxn_m$.

Here the problems are deterministic (`--duration_type`), but you can choose to train
the model on stochastic problems (see section [Stochastic 10x10](#stochastic-10x10)).

The rest of the arguments either define the overall GNN model or the training dynamics.

### Inference

Once a model is trained, you can use it to solve new problems using:
```sh
python3 -m jssp.solve\
    --path "./PATH/TO/EXPERIMENT/"\
    --load_problem "./PATH/TO/INSTANCE.TXT"\
    [--first_machine_id_is_one]  # Optional, if you load taillard problems, you should tell that the index starts at one.
```

Check `jssp/solve.py` to see how to load and use the model in your own python scripts.

## Examples

Unless specified, all training and validation problems are randomly generated.

### Stochastic 10x10

```sh
python3 -m jssp.train\
    --batch_size 245\
    --clip_range 0.20\
    --custom_heuristic_names SPT MWKR MOPNR FDD/MWKR\
    --device cuda:0\
    --duration_type stochastic\
    --ent_coef 0.05\
    --exp_name_appendix stochastic-10x10\
    --fe_type dgl\
    --fixed_validation\
    --gae_lambda 0.99\
    --gamma 1.00\
    --generate_duration_bounds 0.05 0.1\
    --graph_has_relu\
    --graph_pooling max\
    --hidden_dim_actor 32\
    --hidden_dim_critic 32\
    --hidden_dim_features_extractor 64\
    --layer_pooling last\
    --lr 1e-4\
    --max_n_j 100\
    --max_n_m 30\
    --mlp_act gelu\
    --n_epochs 3\
    --n_j 10\
    --n_layers_features_extractor 10\
    --n_m 10\
    --n_mlp_layers_actor 1\
    --n_mlp_layers_critic 1\
    --n_mlp_layers_features_extractor 1\
    --n_steps_episode 9800\
    --n_validation_env 100\
    --n_workers 1\
    --optimizer adamw\
    --ortools_strategy realistic averagistic\
    --residual_gnn\
    --seed 0\
    --target_kl 0.04\
    --total_timesteps 10_000_000\
    --validation_freq 3\
    --vf_coef 2.0
```

### Deterministic 20x20

```sh
python3 -m jssp.train\
    --batch_size 64\
    --clip_range 0.20\
    --custom_heuristic_names SPT MWKR MOPNR FDD/MWKR\
    --device cuda:0\
    --duration_type deterministic\
    --ent_coef 0.05\
    --exp_name_appendix deterministic-20x20\
    --fe_type dgl\
    --fixed_validation\
    --gae_lambda 0.99\
    --gamma 1.00\
    --graph_has_relu\
    --graph_pooling max\
    --hidden_dim_actor 32\
    --hidden_dim_critic 32\
    --hidden_dim_features_extractor 64\
    --layer_pooling last\
    --lr 1e-5\
    --max_n_j 100\
    --max_n_m 30\
    --mlp_act gelu\
    --n_epochs 1\
    --n_j 20\
    --n_layers_features_extractor 10\
    --n_m 20\
    --n_mlp_layers_actor 1\
    --n_mlp_layers_critic 1\
    --n_mlp_layers_features_extractor 1\
    --n_steps_episode 9800\
    --n_validation_env 100\
    --n_workers 1\
    --optimizer adamw\
    --ortools_strategy realistic\
    --residual_gnn\
    --seed 0\
    --target_kl 0.04\
    --total_timesteps 10_000_000\
    --validation_freq 10\
    --vf_coef 2.0
```

The same as previous examples but with a lower batch size so that it fits in memory.
Note that big `--max_n_j` and `--max_n_m` increase the GPU memory requirements.
Also do the validation evaluation every 10 epochs to save some time.


### Stochastic 100x20 with sub sampling

Note: the subsampling feature exists but has not been tested thoroughly.

```sh
python3 -m jssp.train\
    --batch_size 64\
    --clip_range 0.20\
    --custom_heuristic_names SPT MWKR MOPNR FDD/MWKR\
    --device cuda:0\
    --duration_type deterministic\
    --ent_coef 0.05\
    --exp_name_appendix deterministic-subsampling-100x20\
    --fe_type dgl\
    --fixed_validation\
    --gae_lambda 0.99\
    --gamma 1.00\
    --graph_has_relu\
    --graph_pooling max\
    --hidden_dim_actor 32\
    --hidden_dim_critic 32\
    --hidden_dim_features_extractor 64\
    --layer_pooling last\
    --lr 1e-4\
    --max_n_j 100\
    --max_n_m 30\
    --mlp_act gelu\
    --n_epochs 1\
    --n_j 100\
    --n_layers_features_extractor 10\
    --n_m 20\
    --n_mlp_layers_actor 1\
    --n_mlp_layers_critic 1\
    --n_mlp_layers_features_extractor 1\
    --n_steps_episode 9800\
    --n_validation_env 100\
    --n_workers 1\
    --optimizer adamw\
    --ortools_strategy realistic\
    --residual_gnn\
    --sample_n_jobs 10\
    --seed 0\
    --target_kl 0.04\
    --total_timesteps 10_000_000\
    --validate_on_total_data\
    --validation_freq 10\
    --vf_coef 2.0
```

Make use of `--sample_n_jobs` and `--validate_on_total_data`.


### Single fixed problem

```sh
python3 -m jssp.train\
    --batch_size 245\
    --clip_range 0.20\
    --custom_heuristic_names SPT MWKR MOPNR FDD/MWKR\
    --device cuda:0\
    --duration_type deterministic\
    --ent_coef 0.05\
    --exp_name_appendix single-problem\
    --fe_type dgl\
    --first_machine_id_is_one\
    --fixed_problem\
    --fixed_validation\
    --gae_lambda 0.99\
    --gamma 1.00\
    --graph_has_relu\
    --graph_pooling max\
    --hidden_dim_actor 32\
    --hidden_dim_critic 32\
    --hidden_dim_features_extractor 64\
    --layer_pooling last\
    --load_problem instances/taillard/ta57.txt\
    --lr 1e-4\
    --mlp_act gelu\
    --n_epochs 3\
    --n_j 10\
    --n_layers_features_extractor 10\
    --n_m 10\
    --n_mlp_layers_actor 1\
    --n_mlp_layers_critic 1\
    --n_mlp_layers_features_extractor 1\
    --n_steps_episode 9800\
    --n_validation_env 1\
    --n_workers 1\
    --optimizer adamw\
    --ortools_strategy realistic\
    --residual_gnn\
    --seed 0\
    --target_kl 0.04\
    --total_timesteps 10_000_000\
    --vf_coef 2.0
```

Load the taillard problem and use it as the single training and validation instance.
This is useful if you have a specific problem you want to solve. You can set the durations
to be stochastic so that the training instances will be randomly generated around the
durations bound.
