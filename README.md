# JoliJSS2

TODO:
 - Voir si il faut pas rajouter une feature dans les nodes, qui précise quelle machine
 ils utilisent (!!!!!!!!)
 - Implémenter un GIN Features Extractor
 - Implémenter les tests pour env_observation.py et agent_observation.py
 - Ajouter des unit testing de graph embedding et actor_critic

## Différences avec L2D:
 - Ils normalisent les rewards, pas moi
 - Ils fonctionnent en comptant le nombre d'environnements, je compte le nombre de steps
 (et du coup ils s'entrainent à chaque fin d'environnement, moi à chaque fois que j'ai 
 fait n steps)
 - Ils ne fonctionnent pas par batch, moi si
 - Leur loss est le double de la mienne (parce qu'ils ont un policy loss de 2, et que
 je suis obligé d'avoir 1)
