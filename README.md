# JoliJSS2

This repo intends to replicate L2D, a model proposed in this paper:
[Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning](https://arxiv.org/pdf/2010.12367)

It should also provide improvements, in order to apply it to real industry problems

TODO:
 - Implémenter les tests pour env_observation.py et agent_observation.py

## Differences with L2D implementation:
 - Rewards are normalized, I only divide them by a scalar.
 - They update the PPO model every n env run, I do it every n_steps
 - They don't use batching, I do
 - Theyr loss is twice mine.
 - Theyr action space get smaller as tasks are totally completed. Not mine.
 - I add the machine_id in the features
 - The input for actor is [node_embedding, node_embedding, graph_embedding]. For them, 
   it's [node_embedding, graph_embedding]
