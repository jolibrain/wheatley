# Wheatley 

A Job-Shop Scheduling problem (JSSP) solver based on Reinforcement Learning, targeted to solving real-world industrial problems.

## Features
- Trains a scheduler for fixed or problems with uncertainty
- Support for training over random problems and generalize
- Support for training over problems with bounded but uncertain durations
- Reads JSSP in Taillard format, extended for uncertain durations
- Web live training metrics reported with [Visdom]()
- Includes schedule visualization as Gantt charts
- Compares to OR-Tools
- Relies on state-of-the art Deep Learning libraries: written with [Pytorch](), uses [stables-baselines3]() for RL, and [pytorch-geometric]() for graph neural networks

## Installation
```
pip install -r requirements.txt
```

## Run Model:
You can launch a training using this line (make sure to run visdom in another process before):
```
python3 train.py --n_j 6 --n_m 6 --total_timesteps 2000000 --n_validation_env 100 --validation_freq 1000 --n_steps_episode 256 --batch_size 128 --seed 42 --n_workers 8 --insertion_mode choose_forced_insertion --exp_name_appendix "my_run"
```

## Contribute
If you want to contribute to wheatley, make sure to install the pre-commit hooks:
```
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

## Differences with L2D and other JSSP-RL implementations:
 - Rewards are normalized, simply divided by a scalar. This also means that the 
 original implementeation value loss is different than ours
 - L2D updates the PPO model every n environment runs, Wheatley does every n steps, which is more practical
 - Wheatley uses batching
 - Wheatley uses advanced GNN, such as gatv2 thanks to pytorch-geometric
 - Wheatley embeds more information into every node of the schedule graph, yielding more informed policies
 - Wheatley has support for bounded uncertain durations, including at node and reward levels.


