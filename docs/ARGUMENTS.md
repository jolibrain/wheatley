# All Arguments

Here is the list of all arguments available to launch a training.
If an argument expects some specific values, you can get them by
using the `--help` flag.


## JSSP environment

- `--chunk_n_jobs`: sliding window size
- `--duration_type` : either stochastic or deterministic
- `--first_machine_id_is_one` : in pure taillard format, machine numbering start at 1
- `--generate_duration_bounds X Y` : real durations are sampled within the bounds ($-X\%$, $+Y\%$)
- `--load_from_job` : index of first job to use
- `--load_max_jobs`: max number of jobs for sampling
- `--load_problem` :  forces to read a problem definition instead of generating one
- `--max_duration` : max duration of tasks for deterministic problem generation
- `--max_n_j` : max number of jobs (default is  n_j)
- `--max_n_m` : max number of  machines (default is  n_m)
- `--n_j` : number of jobs
- `--n_m` : number of  machines
- `--sample_n_jobs`: number of jobs to sample
- `--validate_on_total_data`: force evaluation on global problem

## RCPSP environment

- `--load_problem` :  forces to read a problem definition instead of generating one
- `--train_dir`: the directory containing all problems you want to train on
- `--test_dir`: the directory containing all test problems
- `--train_test_split`: if no `--test_dir` is provided, the train instances will be splitted according to this ratio

## PPO training

- `--batch_size`: batch size for PPO
- `--clip_range`: clip gradients
- `--dont_normalize_advantage`: do not normlize advantage function
- `--ent_coef`: entropy coefficient in PPO loss
- `--fe_lr`: learning rate of ther feature extreactor, if different from global learning rate
- `--fixed_problem`: force use of same problem along all training
- `--freeze_graph`: freeze graph during learning (for debugging purposes)
- `--gae_lambda`: lambda parameter of the Generalized Advantage Estimation
- `--gamma`: discount factor, default 1 for finite horizon
- `--lr`: learning rate
- `--n_epochs`: number of time a given replay buffer is used during training
- `--n_steps_episode`: number of action per sequence (generally $k \times n_j \times n_m$)
- `--optimizer`: optimizer to use
- `--target_kl`: target kl for PPO
- `--total_timesteps`: total number of training timesteps
- `--vf_coef`: value function coefficient in PPO loss

## Computation efficiency

- `--device`: device id  `cpu`, `cuda:0` ...
- `--n_workers`: number of data collecting threads (size of data buffer is n_steps_episode $\times$ n_workers)
- `--vecenv_type`: type of threading for data collection

## Test and validation options

- `--fixed_random_validation`: number of fixed problem to generate for validation
- `--fixed_validation`: Fix and use same problems for agent evaluation and or-tools. When used, the validation instances are solved once for all baselines (ortools and custom heuristics)
- `--max_time_ortools`: or-tools timeout
- `--n_test_problems`: number of problems to generate for validation (in case they are not pre-generated with fixed_validation and fixed_random_validation
- `--n_validation_env` : number of validation environment for model evaluation
- `--test_print_every`: print frequency of evaluations
- `--validation_freq`: number of steps between evaluations
and their values are reused for the rest of the training. Only the trained model is evaluated every time the validation evaluation is triggered.

## Baseline comparisons

- `--custom_heuristic_names` : heuristics to use as a comparison (`SPT`, `MWKR`, `MOPNR`, `FDD/MWKR`), only available for JSSP problems.
- `--ortools_strategy` : any number of strategies to use for the OR-Tools solver (`realistic`, `averagistic`)

## GNN model

- `--conflicts` : conflict encoding in GNN
- `--graph_has_relu`: add (r)elu in GNN
- `--graph_pooling`:  global pooling mode default is learn (ie pool node)
- `--hidden_dim_actor`: latent dim of actor
- `--hidden_dim_critic`: latent dim of actor
- `--hidden_dim_features_extractor`: latent dimension in GNN
- `--mlp_act`: activation function in MLP in GNN (if any, default to gelu)
- `--n_attention_heads`: number of attention heads in GNN (if any)
- `--n_layers_features_extractor` : number of layers in GNN
- `--n_mlp_layers_actor`: number of layers of actor
- `--n_mlp_layers_critic`: number of layers of critic
- `--n_mlp_layers_features_extractor` : number of layers in MLP in GNN (if any)
- `--normalize_gnn` : add normalization layers in  GNN
- `--residual_gnn` : add residual connections in GNN
- `--reverse_adj_in_gnn` : invert adj direction in pyg feature extractor (for debug, deprecated)

## Modelisation options

- `--do_not_observe_updated_bounds`: task completion time (tct) bounds are computed on-the-fly during trial (necessary for L2D reward model), with this option updated tct bounds are not given to the agent (not observed)
- `--dont_normalize_input`: do not normalize state data
- `--insertion_mode` : allow insertion
- `--observe_duration_when_affect` : with this option, real durations are observed at affectation time and used to tighten task completion time bounds. 
- `--reward_model_config` : reward model
- `--transition_model_config` : transition type

## Model weights loading

- `--reinit_head_before_ppo`: replace the actor and policy heads by newly initialized weights just before starting PPO (and after all model's weights), useful after a pretraining for example
- `--resume`: use the experiment exact name to load the weight of the previous saved model
- `--retrain PATH`: load the model pointed by the PATH

## Others

- `--path`: directory where training logs are saved, a subdirectory is created for each new training
