python -m pp.train_pp --batch_size 1000 --exp_name_appendix pp_FINAL --n_layers_features_extractor 8 --fixed_validation --maze_size_train 6 --maze_size_test 6 --n_epochs 3 --n_steps_episode 10000 --n_workers 10 --path /data1/infantes/networks  --device cuda:$1  --n_train_mazes 100 --n_test_mazes 10 --infinite_dataset     --pp_agent_types 0  

