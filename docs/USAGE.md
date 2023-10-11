# Introduction

Launch the visdom logging server:
```sh
python -m visdom.server
```

Launch a training run:
```sh
python ./train.py
```

Status is displayed  on [localhost](http://localhost:8097).


Commmon options:

- `--n_j` : number of jobs
- `--n_m` : number of  machines
- `--total_timesteps` : total number of training timesteps
- `--n_validation_env` : number of validation environment for model evaluation
- `--n_steps_episode`: number of action per sequence (generally $k \times  n_j \times n_m$)
- `--batch_size` : batch size for PPO
- `--fe_type` :  feature extractor type dgl[default] ou pyg[deprecated] or tokengt[deprecated]
- `--gconv_type` :  convolution type  (for dgl and pyg default: gatv2)
- `--lr` : learning rate
- `--device` : device id  `cpu`, `cuda:0` ...
- `--n_workers` : number of data collecting threads (size of data buffer is n_steps_episode $\times$ n_workers)
- `--features` : state features (default is all)
- `--exp_name_appendix` : print suffix for  visdom


Once a model is trained, you can use it to solve new problems using:
```sh
python3 -m jssp.solve\
    --path "./PATH/TO/EXPERIMENT/"\
    --load_problem "./PATH/TO/INSTANCE.TXT"\
    [--first_machine_id_is_one]  # Optional, if you load taillard problems, you should tell that the index starts at one.
```

See `jssp/solve.py` to see how to load and use the model in your own python scripts.



# Small fixed random problem without uncertainty

```sh
python train.py\
    --n_j 4\
    --n_m 4\
    --total_timesteps 1000000\
    --n_validation_env 1\
    --n_steps_episode 1600\
    --batch_size 160\
    --exp_name_appendix EXAMPLE1\
    --fixed_problem\
    --seed 1
```

- `--fixed_problem` : force use of same problem along all training
- `--seed 1` : force generation of same problem among several trainings


 
# Random problems without uncertainty
```
python train.py --n_j 4 --n_m 4 --total_timesteps 1000000 --n_validation_env 10 --n_steps_episode 1600  --batch_size 1600 --seed 1   --exp_name_appendix EXAMPLE2 
```

Same as above, without `--fixed_problem` 


# Single random problem with uncertainty
```
python train.py --n_j 4 --n_m 4 --total_timesteps 1000000 --n_validation_env 1 --n_steps_episode 1600 --batch_size 1600 --seed 1 --duration_type stochastic --fixed_problem  --reward_model_config Sparse --ortools_strategy averagistic --exp_name_appendix EXAMPLE3 
```


- `--duration_type stochastic` : forces stochastic durations
- `--reward_model_config Sparse` : reward model is based on true execution time, evaluated only on complete schedule
- `--ortools_strategy averagistic` : ortools is given average (or mode) values


# Taillard problem with generated random durations and subsampling of jobs

```
python3 train.py --n_j 50 --n_m 15 --total_timesteps 1000000 --n_validation_env 10 --n_steps_episode 1500 --batch_size 150 --duration_type stochastic --fixed_problem  --load_problem instances/taillard/ta57.txt --first_machine_id_is_one --exp_name_appendix EXAMPLE4 --n_epochs 10  --ortools_strategy averagistic   --device cuda:3 --generate_duration_bounds 0.05 0.1  --load_max_jobs 40 --load_from_job 0
```

- `--load_problem` :  forces to read a problem definition instead of generating one
- `--first_machine_id_is_one` : in pure taillard format, machine numbering start at 1
- `--generate_duration_bounds 0.05 0.1`: generate duration bounds in [v\*0.95, v\*1.1] where v is duration of  fixed loaded problem. 
- `--load_from_job 0` : index of first job to use
- `--load_max_jobs 40` : max number of jobs to use


In the case of randomly generated problem (eg w/o `--load_problem`), instead of `--generate_duration_bounds` one should use:

- `--duration_mode_bounds`
- `--duration_delta` 




# Large problem resolution by sub problem sampling:

## DGL
```
python train.py --n_j 100 --n_m 20 --n_steps_episode 4000 --n_workers 5 --total_timesteps 2000000 --n_validation_env 1 --fixed_validation --fixed_problem --load_problem instances/taillard/ta72.txt --first_machine_id_is_one --n_epochs 10 --n_layers_features_extractor 8 --device cuda:0 --batch_size 200 --exp_name_appendix EXAMPLE5  --sample_n_jobs 10 --validate_on_total_data
```

## TOKENGT
```
python train.py --n_j 100 --n_m 20 --n_steps_episode 4000 --n_workers 2 --total_timesteps 2000000 --n_validation_env 1 --fixed_validation --fixed_problem --load_problem instances/taillard/ta72.txt --first_machine_id_is_one --n_epochs 10 --n_layers_features_extractor 2 --device cuda:0 --batch_size 200 --exp_name_appendix EXAMPLE6  --hidden_dim_features_extractor 128 --fe_type tokengt --conflicts att --hidden_dim_actor 128 --hidden_dim_critic 128  --sample_n _jobs 10 --validate_on_total_data  
```



# All options

## JSSP environment
- `--n_j` : number of jobs
- `--n_m` : number of  machines
- `--max_n_j` : max number of jobs (default is  n_j) [deprecated]
- `--max_n_m` : max number of  machines (default is  n_m) [deprecated]
- `--max_duration` : max duration of tasks for deterministic problem generation
- `--duration_type` : either stochastic or deterministic
- `--generate_duration_bounds X Y` : real durations are sampled within the bounds ($-X\%$, $+Y\%$)
- `--load_problem` :  forces to read a problem definition instead of generating one
- `--first_machine_id_is_one` : in pure taillard format, machine numbering start at 1
- `--load_from_job` : index of first job to use
- `--load_max_jobs` : max number of jobs to use

## PPO training
- `--lr` : learning rate
- `--gamma` : discount factor, default 1 for finite horizon
- `--clip_range`: clip gradients
- `--target_kl`: target kl for PPO
- `--ent_coef`: entropy coefficient in PPO loss
- `--vf_coef`: value function coefficient in PPO loss
- `--dont_normalize_advantage`: do not normlize advantage function
- `--fe_lr` : learning rate of ther feature extreactor, if different from global learning rate
- `--n_epochs` : number of time a given replay buffer is used during training
- `--optimizer`: optimizer to use
- `--freeze_graph` : freeze graph during learning (for debugging purposes)
- `--total_timesteps` : total number of training timesteps
- `--n_steps_episode`: number of action per sequence (generally $k \times  n_j \times n_m$)
- `--batch_size` : batch size for PPO
- `--fixed_problem` : force use of same problem along all training

## Computation efficiency
- `--device` : device id  `cpu`, `cuda:0` ...
- `--n_workers` : number of data collecting threads (size of data buffer is n_steps_episode $\times$ n_workers)
- `--vecenv_type` : type of threading for data collection

## Test and validation options
- `--n_validation_env` : number of validation environment for model evaluation
- `--fixed_validation`: fix and use same problems for agent evaluation and or-tools
- `--fixed_random_validation`: number of fixed problem to generate for validation
- `--validation_freq`: number of steps between evaluations
- `--max_time_ortools`: or-tools timeout
- `--n_test_problems`: number of problems to generate for validation (in case they are not pre-generated with fixed_validation and fixed_random_validation
- `--test_print_every`: print frequency of evaluations

## Baseline comparisons
- `--ortools_strategy` : any number of strategies to use for the OR-Tools solver (`realistic`, `averagistic`)
- `--custom_heuristic_names` : heuristics to use as a comparison (`SPT`, `MWKR`, `MOPNR`, `FDD/MWKR`)

## GNN model
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

## Actor and critic network parameters
- `--n_mlp_layers_actor`: number of layers of actor
- `--hidden_dim_actor`: latent dim of actor
- `--n_mlp_layers_critic`: number of layers of critic
- `--hidden_dim_critic`: latent dim of actor

## Model Options
- `--transition_model_config` : transition type
- `--insertion_mode` : allow insertion
- `--reward_model_config` : reward model
- `--dont_normalize_input`: do not normalize state data
- `--observe_duration_when_affect` : with this option, real durations are observed at affectation time and used to tighten task completion time bounds. 
- `--do_not_observe_updated_bounds`: task completion time (tct) bounds are computed on-the-fly during trial (necessary for L2D reward model), with this option updated tct bounds are not given to the agent (not observed)

## Sub problem sampling
- `--load_from_job` : start index for sub problem sampling
- `--load_max_jobs` : max number of jobs for sampling
- `--sample_n_jobs` : number of jobs to sample
- `--chunk_n_jobs` : sliding window size
- `--validate_on_total_data` : force evaluation on global problem

## Model weights loading
- `--resume` : use the experiment exact name to load the weight of the previous saved model
- `--retrain PATH` : load the model pointed by the PATH
- `--reinit_head_before_ppo` : replace the actor and policy heads by newly initialized weights just before starting PPO (and after all model's weights), useful after a pretraining for example

## Others
- `--path` : directory where training logs are saved, a subdirectory is created for each new training

# Real example

Training a GNN on 10x10 random stochastic instances from scratch :
```sh
python3 train.py\
    --n_j 10\
    --n_m 10\
    --max_n_j 20\
    --max_n_m 20\
    --lr 1e-5\
    --ent_coef 0.01\
    --vf_coef 2.0\
    --target_kl 0.04\
    --clip_range 0.20\
    --gamma 0.99\
    --gae_lambda 0.95\
    --optimizer adamw\
    --fe_type dgl\
    --residual_gnn\
    --graph_has_relu\
    --graph_pooling learn\
    --hidden_dim_features_extractor 64\
    --n_layers_features_extractor 10\
    --mlp_act gelu\
    --layer_pooling last\
    --n_mlp_layers_features_extractor 1\
    --n_mlp_layers_actor 1\
    --n_mlp_layers_critic 1\
    --hidden_dim_actor 32\
    --hidden_dim_critic 32\
    --total_timesteps 10000000\
    --n_validation_env 100\
    --n_steps_episode 4900\
    --batch_size 245\
    --n_epochs 3\
    --fixed_validation\
    --custom_heuristic_names SPT MWKR MOPNR FDD/MWKR\
    --seed 0\
    --duration_type stochastic\
    --generate_duration_bounds 0.05 0.1\
    --ortools_strategy averagistic realistic\
    --device cuda:3\
    --n_workers 10\
    --exp_name_appendix random-instances_from-scratch
```
