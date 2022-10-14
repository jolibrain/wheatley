# Wheatley / MuZero 


## Installation
```
git submodule update --init
pip install -r requirements.txt -r muzero_general/requirements.txt
```


## Configuration MuZero
Les réglages spécifiques à MuZero se font dans le fichier `games/wheatley.py`, notamment le pourcentage d'injection de solutions OR-Tools via le paramètre `self.ortools_ratio` :
- une valeur de 0 laisse MuZero explorer par lui même
- une valeur de 1 ne montre à MuZero que des solutions issues d'OR-Tools


## Exemple de training 12x16
```
RAY_TMPDIR=~/ray python3 train_muzero.py --n_j 12 --n_m 16 --n_validation_env 10 --duration_type stochastic --fixed_problem --reward_model_config optimistic --ortools_strategy averagistic --load_problem instances/agilea/small_12_unc.txt
```


## Exemple de training 64x16 avec DGL et 5% d'incertitude
```
RAY_TMPDIR=~/ray python3 train_muzero.py --n_j 64 --n_m 16 --n_validation_env 10 --duration_type stochastic --fixed_problem --reward_model_config optimistic --ortools_strategy averagistic --load_problem instances/agilea/3B-OF_en_cours_ou_dispo_16_colonnes.txt --load_from_job 0 --load_max_jobs 64 --generate_duration_bounds 0.05 --max_edges_upper_bound_factor 2 --features duration --fe_type dgl --graph_pooling learn --conflicts clique
```


## Ray
Pour stopper tous les processus lancés par MuZero il peut être nécessaire d'exécuter la commande suivante :
```
ray stop --force
```
