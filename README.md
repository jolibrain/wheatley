# Wheatley 

This repo intends to replicate L2D, a model proposed in this paper:
[Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning](https://arxiv.org/pdf/2010.12367)

It should also provide improvements, in order to apply it to real industry problems

## Differences with L2D implementation:
 - Rewards are normalized, I only divide them by a scalar. This also means that the 
 value they use for value loss is not the same as mine.
 - Theyr loss is twice mine (at least for ent_coef and pg_coef).
 - They update the PPO model every n env run, I do it every n_steps
 - They don't use batching, I do
 - The input for actor is [node_embedding, node_embedding, graph_embedding]. For them, 
   it's [node_embedding, graph_embedding]

## Questions:
 - Due to the graph embedding process, first nodes don't have access to the
 information contained in the later nodes... It means the model can't make a decision
 based on the whole information at the beginning. The only way to access this 
 information is through the graph embedding, which is diffuse... I think that the
 model could perform better with more information.
