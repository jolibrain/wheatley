#!/bin/sh

# Example launching script to benchmark a 10x10 stochastic model.
# You should replace the args with the args of the model you want to evaluate.
# The args `n_j` and `n_m` are useless here.

python3 eval.py\
    --max_n_j 20\
    --max_n_m 20\
    --lr 0.0001\
    --ent_coef 0.05\
    --vf_coef 1.0\
    --target_kl 0.1\
    --clip_range 0.25\
    --gamma 1.0\
    --gae_lambda 0.99\
    --optimizer lion\
    --fe_type dgl\
    --residual_gnn\
    --graph_has_relu\
    --graph_pooling learn\
    --hidden_dim_features_extractor 24\
    --n_layers_features_extractor 5\
    --mlp_act gelu\
    --layer_pooling last\
    --n_mlp_layers_features_extractor 1\
    --n_mlp_layers_actor 2\
    --n_mlp_layers_critic 1\
    --hidden_dim_actor 16\
    --hidden_dim_critic 16\
    --total_timesteps 100000000\
    --n_validation_env 10\
    --n_steps_episode 4900\
    --batch_size 245\
    --n_epochs 3\
    --fixed_validation\
    --custom_heuristic_names SPT MWKR MOPNR FDD/MWKR\
    --seed 0\
    --device cuda:0\
    --n_workers 1\
    --path "./saved_networks/10j10m_Dstochastic_Tsimple_RSparse_GNNdgl_CONVgatv2_POOLlearn_L5_HD24_H4_Cclique_benchmark-stochastic/"
