# Advice
Reinforcement learning (RL) outcomes are strongly influenced by the careful selection of hyperparameters.
The Wheatley project is no exception.
This guide aims to offer insights regarding potential hyperparameter values suitable for your experiments.

It's worth noting that a significant challenge with RL is its sample efficiency.
Often, it may take an extended period to observe the implications of a particular hyperparameter choice.
Even though the rate of improvement may diminish over time, training typically remains stable, allowing the model to enhance its performance progressively.

While our expertise doesn't primarily lie in RL, you might possess a deeper understanding of the impacts of hyperparameters on performance.
Please consider our advice with this in mind.

## Entropy
The entropy coefficient plays a crucial role in promoting exploration within the model, helping to prevent stagnation at local minima.
Generally, values within the range $[0.005, 0.05]$ are effective.
We use $0.05$ for every JSSP experiments.

Don't be afraid if the entropy loss dominates the overall losses.
The model is evaluated using the argmax of its output probabilities, so as long as the model is able to tell which action is preferred it will be enough.

Exploration is important.

## Model size
Given that we employ a Graph Neural Network (GNN), the receptive field of the model depends on the number of GNN layers (`--n_layers_features_extractor`).
For larger instances, an increased number of layers is recommended.
However, the width of the model (`--hidden_dim_features_extractor`) can remain relatively modest, typically between $32$ and $64$.

Remember, larger models can be more challenging and time-consuming to train, particularly in the realm of RL.
Interestingly, for achieving optimal generalization, models often require more layers than they do when trained on smaller instances.

## Training times
Below is a general outline of the training durations required for instances of varying sizes:

|  Size  |   Time   |
|:------:|----------|
|  6x6   | 4 hours  |
| 10x10  | 24 hours |
| 15x15  | 72 hours |
| 20x20  | 1 week   |

You can save some time during evaluation by using the `--validation_freq` flag.

## When should you stop the training?
Monitor the ep_rew_mean curve closely. This metric tends to show continuous improvement, even if the validation mean makespan remains static.

It's essential to understand that as the model approaches the performance levels of techniques like OR-tools and other heuristics, the pace of enhancement slows considerably.

Should you find the model plateauing, consider reducing the entropy coefficient as a first step before resuming training.
If the stagnation persists, you can try a new training session with an bigger model.

## Stochastic vs deterministic
The gap between OR-Tools and heuristics solutions is getting lower when tackling stochastic problems.
This is because those solvers are not tailored for stochastic problems so they end up with a pool of solutions that performe all the same.

This means that for PPO it is easier to catch up and to be better than those solvers.
