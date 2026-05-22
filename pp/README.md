# Exemples de scripts d'apprentissage :

version chrono ../pp.sh NUM_GPU SIZE BATCH_SIZE (pour size 10, BATCHSIZE DE 512 à 1024 passe)
version non chrono avec waypoints wp: ../pp_wp.sh NUM_GPU SIZE BATCH_SIZE
version non chrono avec waypoint + removal de WP: ../pp_wpr.sh NUM_GPU SIZE BATCH_SIZE
version non chrono path complet : ../pp_path.sh NUM_GPU SIZE BATCH_SIZE
version chrono sur GNN hierarchique : ../pp_hier.sh
exemple de resume sur non chrono path: ../pp_path_resume.sh

Attention ! il faut changer dans les scripts les path du modèle --path et le path de stockages des observations --store_rollouts_on_disk

# Exemples de scripts d'évaluation:

../launch_all.sh pour la version chrono
ça sort les stats brutes, pas mise en forme.
../launch_hard.sh lance les evals sur les pb maze-hard TRES LONG !!!


# TODO :

- Faire l'apprentissage sur des tailles plus grandes que 10x10 pour voir si ça généralise mieux sur 30x30 et sur maze-hard

- retester avec attributs positionnels ? RWPE et LAPPE, pas testés depuis un moment, vérifier dans les node_embedders

- Lookahead sur chrono ?
- Faire marcher path?


- Finir BNPool (-> Guillaume) https://arxiv.org/abs/2501.09821
- Regarder maxCutPooling  (-> Guillaume) https://arxiv.org/abs/2409.05100