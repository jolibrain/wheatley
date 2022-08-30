# Généralités

wheatly est lancé via la commande `python train.py` , il affiche son état courant en [local](http://localhost:8097). Il a besoin d'un grand nombres d'options, détaillées dans `args.py`, avec leurs valeurs par défaut. Les plus importantes sont:

- `--n_j` : nombre de jobs
- `--n_m` : nombre maximal de machines
- `--total_timesteps` : nombre total d'actions à exécuter durant l'entrainement 
- `--n_validation_env` : nombre d'environnements pour évaluer la politique (pour faire une moyenne, dans le cas de problème avec incertitudes)
- `--n_steps_episode`: nombre d'actions par séquence (en général n_j $\times$ n_m)
- `--batch_size 360` : nombre de séquences à traiter en parallèle lors de l'optimisation
- `--gconv_type` : type de convolution dans le graph neural network
- `-- lr` : learning rate
- `--device` : device sur lequel faire les calculs parmi `cpu`, `cuda:0` ...
- `--n_workers` : nombre d'environnements de génération des traces en parallèle (le nombre total de traces par itération sera de total_timesteps $\times$  n_workers)
- `--max_edges_upper_bound_factor`: nombre maximal d'arcs : 2 pour deux fois plus d'arcs que de noeuds 
- `--features` : features à incorporer dans l'état en plus du pur graphe de précédences 



# Example de petit problème (au hasard) sans incertitude

```
python3 train.py --n_j 6 --n_m 6 --total_timesteps 1000000 --n_validation_env 1 --n_steps_episode 360 --batch_size 360 --seed 1 --gconv_type gatv2 --fixed_problem --lr 0.0002
```
ici on utilise les options générales, et on utilise toujours le même problème généré en posant l'option `--fixed_problem`. De plus, d'un run à l'autre, on génère le même problème en fixant la graine de génération de nombre aléatoires `--seed 1`. 


# Exemple de problème 'taillard'
```
python3 train.py --n_j 100 --n_m 20 --n_steps_episode 2000 \
        --total_timesteps 10000000 --n_validation_env 1 \
        --fixed_problem --load_problem instances/agilea/cas_modele_simulation.txt \
        --lr 0.00002 \
        --device cuda:0 --n_workers $(nproc) --batch_size 24 --max_edges_upper_bound_factor 2
```

Ici encore, on utilise toujours le même problème fixe, mais il est spécifié par l'option `--load_problem instances/agilea/cas_modele_simulation.txt`. (c'est si aucun problème n'est spécifié qu'un est généré comme dan l'exemple précédent. 

# Exemple d'apprentissage sur plusieurs problèmes (générés au hasard), sans incertitudes
```
python3 train.py --n_j 6 --n_m 6 --total_timesteps 1000000 --n_validation_env 1 --n_steps_episode 360 --batch_size 360 --seed 1 --gconv_type gatv2 --lr 0.0002
```

Si on enlève l'option `--fixed_problem`, des problèmes sont générés à chaque nouvel boucle de l'apprentissage. 


# Exemple d'apprentissage sur un seul problème (généré au hasard), avec des incertiudes
```
python3 train.py --n_j 6 --n_m 6 --total_timesteps 1000000 --n_validation_env 1 -n_steps_episode 360 --batch_size 360 --seed 1 --gconv_type gatv2 --lr 0.0002 --duration_type stochastic --fixed_problem --features duration --reward_model_config optimistic --ortools_strategy averagistic
```

Ici, on ajoute les options:

- `--duration_type stochastic` : pour dire que les durées ont des incertitudes
- `--reward_model_config optimistic` : fixe les récompenses intermédiaires sur les valeurs au plus tôt de la fin des tâches
- `--ortools_strategy averagistic` : donne les valeurs moyennes (de mode plus exactement) de durée à ortools comme valeur fixes. 

        
# Example de problème réel avec génération d'incertitudes à plus ou moins 5% et sampling des problèmes parmi l'ensemble
```
python3 train.py --n_j 64 --n_m 16 --total_timesteps 1000000 --n_validation_env 10 --n_steps_episode 1024 --batch_size 60 --duration_type stochastic --fixed_problem --lr 0.00001 --load_problem instances/agilea/3B-OF_en_cours_ou_dispo_16_colonnes.txt --exp_name_appendix from_0_max_64 --n_epochs 3 --reward_model_config optimistic --ortools_strategy averagistic --load_from_job 0 --load_max_jobs 64 --n_workers 4 --device cuda:0 --generate_duration_bounds 0.05 --max_edges_upper_bound_factor 2 --validation_batch_size 10
```

les principale nouvelle options sont:

- `--generate_duration_bounds 0.05`: génère des bornes à  plus ou moins 5% en chargeant un problème fixé. Dans le cas d'un problème avec incertitudes généré aléatoirement, ce sont les options duration_mode_bounds et  duration_delta qui servent à générer les durées.
- `--load_from_job 0` : indice du premier job à utiliser
- `--load_max_jobs 64` : nombre max de jobs à échantillonner

# Autres options

- `--max_duration` : durée max des tâches, pour générer un problème déterministe
- `--max_n_j` : nombre max de jobs, en général égal à n_j
- `--max_n_m` : nombre max de machines, en général égal à n_m
- `--path` : endroit où sauver le réseau appris
- `--exp_name_appendix` : suffixe d'ffichage dans visdom
- `--vecenv_type` : type de parallélisation de la collecte des données
- `--n_epochs` : nombre de passes d'optimisation sur un même ensemble de données
- `--fe_lr` : learning rate du graph neural net, si différent du learning rate global
- `--optimizer`: alogrithme d'optimisation à utliser
- `--freeze_graph` : gel du graph neural net (pour debug)
- `--custom_heuristic_name`: évaluation d'une heuristique simple en plus
- `--retrain` : mettre pour repartir d'un ancien réseau

## Options de test et validation 

- `--fixed_validation`: force l'utilisation des mêmes problèmes pour l'agent et OR-tools lors de l'évaluation
- `--fixed_random_validation`: nombre de problèmes fixes à générer pour moyenner
- `--validation_freq`: nombre d'étapes entre deux évaluations
- `--max_time_ortools`: durée max données à ORtools pour résoudre le problème
- `--validation_batch_size`: batch size lors de la validation
- `--n_test_problems`: nombre de problèmes à générer lors des évalutions (cas où ils ne sont pas fixés uen fois pour toute au début)
- `--test_print_every`: fréquence d'affichages des évaluations

## Options de PPO

- `--gamma` : discount factor, laisser un pour un horizon fini
- `--clip_range`: amplitude max des gradients
- `--target_kl`: distance max entre les politiques
- `--ent_coef`: coefficient à appliquer à la loss sur l'entropie
- `--vf_coef`: coefficient à appliquer à la loss du critique
- `--dont_normalize_advantage`: ne pas normaliser lors du calcul du gain espérés

## Options du Graph neural net:

- `--graph_pooling`: type de pooling global du graphe
- `--mlp_act`: activation dans les MLP
- `--graph_has_relu`: ajout de relu dans le graphe
- `--n_mlp_layers_features_extractor` : nombre de couches des MLP graph neural net
- `--n_layers_features_extractor` : nombre de couches  graph neural net
- `--hidden_dim_features_extractor`: largeur des données dans le graphe
- `--n_attention_heads`: nombre de tête d'attention pour les convolutions basées attention
- `--reverse_adj_in_gnn` : inverser le sens des adjacences dans le graphe (pour debug)
- `--residual_gnn` : ajout de connections residuelles dans le GNN
- `--normalize_gnn` : ajout de normalisation dans le GNN
- `--conflicts_edges` : remplacer les attributs de conflits par des arcs

## Paramètres de l'acteur and Critic network parameters
- `--n_mlp_layers_shared` : nombre de couches partagées entre l'acteur et le critique
- `--hidden_dim_shared`: largeur des couches partagées entre l'acteur et le critique
- `--n_mlp_layers_actor`: nombre de couches de l'acteur
- `--hidden_dim_actor`: largeur des couches de l'acteur
-  `--n_mlp_layers_critic`: nombre de couches du critique
- `--hidden_dim_critic`: largeur des couches du critique

## Model Options
- `--transition_model_config` : type de transition
- `--insertion_mode` : type d'insertions possibles (sous type des transitions)
- `--reward_model_config` : type de récompenses
- `--dont_normalize_input`: normalise ou non les attributs des noeuds du graphe en entrée

## Echantiollonage de sous problème pour l'apprentissage: : 
- `--load_from_job` : indice de départ pour l'échantillonage de sous problème 
- `--load_max_jobs` : nombre max de jobs à charger
- `--sample_n_jobs` : nombre de jobs à échantillonner
- `--chunk_n_jobs` : taille des fenêtre glissante pour l'échantillonage


