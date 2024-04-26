# Wheatley

A Job-Shop Scheduling problem (JSSP) and Ressource-Constrained Planning Scheduling Problem (RCPSP) solver based on Reinforcement Learning, targeted to solving real-world industrial problems, and more.

This repo contains the official implementation of [Learning to Solve Job Shop Scheduling under Uncertainty](https://arxiv.org/abs/2404.01308)

## Features

- Trains a scheduler for fixed or problems with uncertainty
- Support for training over random problems and generalize
- Support for training over problems with bounded but uncertain durations
- Reads JSSP in Taillard format, extended for uncertain durations
- Web live training metrics reported with [Visdom](https://ai.facebook.com/tools/visdom/)
- Includes schedule visualization as Gantt charts
- Compares to OR-Tools
- Relies on state-of-the art Deep Learning libraries: written with [Pytorch](https://pytorch.org/),  and [DGL](https://www.dgl.ai/) for graph neural networks

## Installation

- Install pytorch for your hardware [(instructions here)](https://pytorch.org/get-started/locally/)
- Install dgl for your hardware [(instructions here)](https://www.dgl.ai/pages/start.html)
- Install other dependencies: `pip install -r requirements.txt`

Note: for windows users, we strongly recommend to use [anaconda](https://www.anaconda.com/)

## How to use

See [JSSP](docs/JSSP.md), [PSP](/docs/PSP.md), [ARGUMENTS](/docs/ARGUMENTS.md) and [ADVICE](/docs/ADVICE.md) for more information.

## Contribute

If you want to contribute to wheatley, make sure to install the pre-commit hooks:

```sh
pre-commit install
```

## Technical details

- Wheatley learns how to schedule well and generalize over problems and/or uncertainty. It works from a representation of the schedule state-space directly, as opposed to the state-space of jobs and machines.
- Uses PPO as the main RL algorithm
- Captures schedules in the form of graphs and trains with an underlying Graph Neural Network
- Large number of hyper-parameters, default values are set to the best currently known values
- A small choice of different rewards is implemented.

## References

- Wheatley first intended to replicate L2D, a model proposed in this paper:
[Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning](https://arxiv.org/pdf/2010.12367)
- Uses some intuitions and ideas from [A Reinforcement Learning Environment For Job-Shop Scheduling](https://arxiv.org/abs/2104.03760)

## Differences with L2D and Other JSSP-RL Implementations

- Rewards are normalized
- Wheatley uses proper batching and parallel environments
- Wheatley uses advanced GNN, such as gatv2  (with edge info) thanks to DGL.
- Wheatley embeds more information into every node of the schedule graph (like propagated time bounds), yielding more informed policies
- Wheatley has support for bounded uncertain durations, including at node and reward levels.

## Citing Wheatley (JSSP)
```
@inproceedings{wheatley-jssp,
  title={Learning to Solve Job Shop Scheduling under Uncertainty},
  author={Guillaume Infantes and St/'ephanie Roussel and Pierre Pereira and Antoine Jacquet and Emmanuel Benazera},
  booktitle={21th International Conference on Integration of Constraint Programming, Artificial Intelligence, and Operations Research (CPAIOR)},
  year={2024},
}

```
