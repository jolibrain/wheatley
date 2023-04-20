




# Benchmark launch line
(problems are generated with same parameters as taillard, ie uniform duration in [1,99]):
```
python train.py --n_j 15 --n_m 15 --n_steps_episode 5625 --n_workers 10 --total_timesteps 2000000 --n_validation_env 100 --fixed_validation  --lr 0.0002  --n_epochs 3 --n_layers_features_extractor 8  --batch_size 100 --exp_name_appendix 15x15_L2D  --hidden_dim_features_extractor 64  --max_time_ortools 3600
```

# Results on taillard problems, compared to [L2D](https://github.com/zcaicaros/L2D).

## Methodology

- Problems are generated randomly using taillard parameters (ie duration is uniformly sampled in [1,99]).

- 100 problems are generated first and then used for every evaluation. 

- At every iteration, new problems are generated randomly and used to train wheatley. 

- This shows geenralization abilities, as train problems are extremely unlikely to be the same as evaluation problems. 

- This is the same setup as L2D. 

## Results 

- Obj. is average makespan

- Gap is percent above optimal

- Time is inference time (ie time to obtain decision, not considering training time)


|  size   | criterion | L2D     | Wheatley |
| :-:     | :-:       |  --:    |  --:     |
| 6 x 6   | Obj.      | 574.09  |   532.98 |
| 6 x 6   | Gap(%)    | 17.7  |   9.1  |
| 10 x 10 |  Obj.     | 988.58  | 920.09   |
| 10 x 10 | Gap(%)     | 22.3  |  14.9  |
| 15 x 15 | Obj.      | 1505.79 | 1414.54  |
| 15 x 15 | Gap(%)    | 26.7  | 17.7   |
| 20 x 20 | Obj.      | 2007.76 | 1930.94  |
| 20 x 20 | Gap(%)       | 29.0  | 24.0   |
| 30 x 20 | Obj.      | 2508.27 | 2410.16  |
| 30 x 20 | Gap(%)       | 29.2  | 24.1   |
      


# Indicative size and computational costs:

For an internal hidden dimension of 64, 8 graph convolution layers, the GNN has 1.7M parameters.

The corresponding FLOPs depending on problem size is:

| size    |  FLOPs |
|:-------:|-------:|
| 6 x 6   |  0.2G  |
| 10 x 10 |  0.7G  |
| 15 x 15 |  0.9G  |
| 20 x 20 |  3.7G  |
| 30 x 20 |  7.1G  |
