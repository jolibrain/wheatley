# Solving JSSP problems

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

## All arguments

### JSSP environment
- `--n_j` : number of jobs
- `--n_m` : number of  machines
- `--max_n_j` : max number of jobs (default is  n_j)
- `--max_n_m` : max number of  machines (default is  n_m)
- `--max_duration` : max duration of tasks for deterministic problem generation
- `--duration_type` : either stochastic or deterministic
- `--generate_duration_bounds X Y` : real durations are sampled within the bounds ($-X\%$, $+Y\%$)
- `--load_problem` :  forces to read a problem definition instead of generating one
- `--first_machine_id_is_one` : in pure taillard format, machine numbering start at 1
- `--load_from_job` : index of first job to use
- `--load_max_jobs` : max number of jobs to use

### PPO training
- `--lr`: learning rate
- `--gamma`: discount factor, default 1 for finite horizon
- `--gae_lambda`: lambda parameter of the Generalized Advantage Estimation
- `--clip_range`: clip gradients
- `--target_kl`: target kl for PPO
- `--ent_coef`: entropy coefficient in PPO loss
- `--vf_coef`: value function coefficient in PPO loss
- `--dont_normalize_advantage`: do not normlize advantage function
- `--fe_lr`: learning rate of ther feature extreactor, if different from global learning rate
- `--n_epochs`: number of time a given replay buffer is used during training
- `--optimizer`: optimizer to use
- `--freeze_graph`: freeze graph during learning (for debugging purposes)
- `--total_timesteps`: total number of training timesteps
- `--n_steps_episode`: number of action per sequence (generally $k \times n_j \times n_m$)
- `--batch_size`: batch size for PPO
- `--fixed_problem`: force use of same problem along all training

### Computation efficiency
- `--device`: device id  `cpu`, `cuda:0` ...
- `--n_workers`: number of data collecting threads (size of data buffer is n_steps_episode $\times$ n_workers)
- `--vecenv_type`: type of threading for data collection

### Test and validation options
- `--n_validation_env` : number of validation environment for model evaluation
- `--fixed_validation`: Fix and use same problems for agent evaluation and or-tools. When used, the validation instances are solved once for all baselines (ortools and custom heuristics)
and their values are reused for the rest of the training. Only the trained model is evaluated every time the validation evaluation is triggered.
- `--fixed_random_validation`: number of fixed problem to generate for validation
- `--validation_freq`: number of steps between evaluations
- `--max_time_ortools`: or-tools timeout
- `--n_test_problems`: number of problems to generate for validation (in case they are not pre-generated with fixed_validation and fixed_random_validation
- `--test_print_every`: print frequency of evaluations

### Baseline comparisons
- `--ortools_strategy` : any number of strategies to use for the OR-Tools solver (`realistic`, `averagistic`)
- `--custom_heuristic_names` : heuristics to use as a comparison (`SPT`, `MWKR`, `MOPNR`, `FDD/MWKR`)

### GNN model
- `--graph_pooling`:  global pooling mode default is learn (ie pool node)
- `--mlp_act`: activation function in MLP in GNN (if any, default to gelu)
- `--graph_has_relu`: add (r)elu in GNN
- `--n_mlp_layers_features_extractor` : number of layers in MLP in GNN (if any)
- `--n_layers_features_extractor` : number of layers in GNN
- `--hidden_dim_features_extractor`: latent dimension in GNN
- `--n_attention_heads`: number of attention heads in GNN (if any)
- `--reverse_adj_in_gnn` : invert adj direction in pyg feature extractor (for debug, deprecated)
- `--residual_gnn` : add residual connections in GNN
- `--normalize_gnn` : add normalization layers in  GNN
- `--conflicts` : conflict encoding in GNN
- `--n_mlp_layers_actor`: number of layers of actor
- `--hidden_dim_actor`: latent dim of actor
- `--n_mlp_layers_critic`: number of layers of critic
- `--hidden_dim_critic`: latent dim of actor

### Modelisation options
- `--transition_model_config` : transition type
- `--insertion_mode` : allow insertion
- `--reward_model_config` : reward model
- `--dont_normalize_input`: do not normalize state data
- `--observe_duration_when_affect` : with this option, real durations are observed at affectation time and used to tighten task completion time bounds. 
- `--do_not_observe_updated_bounds`: task completion time (tct) bounds are computed on-the-fly during trial (necessary for L2D reward model), with this option updated tct bounds are not given to the agent (not observed)

### Sub problem sampling
- `--load_from_job`: start index for sub problem sampling
- `--load_max_jobs`: max number of jobs for sampling
- `--sample_n_jobs`: number of jobs to sample
- `--chunk_n_jobs`: sliding window size
- `--validate_on_total_data`: force evaluation on global problem

### Model weights loading
- `--resume`: use the experiment exact name to load the weight of the previous saved model
- `--retrain PATH`: load the model pointed by the PATH
- `--reinit_head_before_ppo`: replace the actor and policy heads by newly initialized weights just before starting PPO (and after all model's weights), useful after a pretraining for example

### Others
- `--path`: directory where training logs are saved, a subdirectory is created for each new training

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
