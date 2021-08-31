# Classes specification

This document intend to define clear specification for the classes that will be used in
any way by the main.py script.

## Problem description

```python
# There are 2 constructors here : 1 for specific problems, and 1 for problem classes
class ProblemDescription:
    __init__(self, 
             n_jobs: int, 
             n_machines: int, 
             max_duration: int,
             transition_model_config: Dict,
             reward_model_config: Dict
             ) -> ProblemDescription
    __init__(self, 
             affectations: numpy.array, 
             durations: numpy.array,
             transition_model_config: Dict,
             reward_model_config: Dict
             ) -> ProblemDescription
```

## Solution

```python
class Solution:
    def __init__(self, schedule: numpy.array) -> Solution
```

## Agent 

This part comprises the Agent interface and all the engine going in the back. It is 
is mostly handled by stable baselines 3, and especially by the PPO implementation of it.

```python
class Agent:
    model: stable_baselines3.common.BaseAlgorithm
    __init__(self, env: Env, model: PPO = None) -> Agent
    save(self, path: str) -> None
    @classmethod
    load(cls, path: str, env: Env) -> Agent
    # Loads an ancientely trained agent, to allow using pretrained agents
    train(self, problem_description: ProblemDescription, total_timesteps: int) -> None
    # Trains the agent on a specific problem for n steps
    predict(self, problem_description: ProblemDescription) -> Solution
    # Gives the solution found by running the agent and env until termination 
```

```python
# This is only a subpart of the spec of stable_baselines3.ppo.PPO
class stable_baselines3.common.BaseAlgorithm:
    policy: Policy
    env: Env
    __init__(self, 
             policy: stable_baselines3.common.policies.ActorCriticPolicy,
             env: gym.Env,
             policy_kwargs: Dict[
                str, stable_baselines3.common.torch_layers.BaseFeaturesExtractor
             ]
             ) -> stable_baselines3.ppo.PPO
    predict(observation: Dict[str, Union[int, numpy.array]]) -> Tuple[int, Dict]
    # This should return an action for every possible state of the given environment 
    learn(total_timesteps: int) -> None
    # Trains the agent on the internal env
```

```python
class Policy(stable_baselines3.common.policies.ActorCriticPolicy):
    mlp_extractor: MLPExtractor
    features_extractor: FeaturesExtractor
    __init__(self, **kwargs) -> Policy
    forward(self, observation: Dict, deterministic: bool) -> int, int, torch.Tensor
```

```python
class FeaturesExtractor(stable_baselines3.common.torch_layers.BaseFeaturesExtractor):
    # List of torch_geometric.nn.conv.MessagePassing features extractors, built on MLPs
    features_extractors: torch.nn.ModuleList 
    __init__(self, 
             observation_space: gym.spaces.Dict
             ) -> FeaturesExtractor
    forward(self, 
            observation: Dict[str, Union[int, torch.Tensor]]
            ) -> torch.Tensor
```

```python
class GINFeaturesExtractor(FeaturesExtractor):
   # The features_extractors are torch_geometric.nn.conv.GINConv
   features_extractors: torch.nn.ModuleList 
```

```python
class MLPExtractor(torch.nn.Module):
    actor: MLP
    critic: MLP
    latent_dim_pi: int  # Necessary for stable_baselines3 API
    latent_dim_vf: int  # Necessary for stable_baselines3 API
    __init__(self) -> ActorCritic
    forward(self, extracted_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

```python
class MLP(torch.nn.Module):
    __init__(self,
             n_layers: int,
             input_dim: int,
             hidden_dim: int,
             output_dim: int,
             batch_norm: bool,
             device: torch.device
             ) -> MLP
    forward(self, x: torch.Tensor) -> torch.Tensor
```

## Env

```python
class Env(gym.Env):
    transition_model: TransitionModel
    reward_model: RewardModel
    __init__(self,
             problem_description: ProblemDescription
             ) -> Env
    step(self, action: int) -> Tuple[Dict, int, bool, Dict]
    # Necessary for the gym.Env interface
    reset(self) -> Dict
    # Necessary for the gym.Env interface
    get_solution(self) -> Solution
    # Returns the found solution if env is done. Else, returns False
```

```python
class TransitionModel(abc.ABC):
    state: State
    __init__(self, 
             affectations: numpy.array, 
             durations: numpy.array, 
             node_encoding: str
             ) -> TransitionModel
    @abstractmethod
    run(self, action: int) -> None
    # Computes and apply the transition defined by choosing action in current state
    get_observation(self) -> Dict
    # Calls state.to_torch_geometric()
    reset(self) -> None
    # Calls state.reset()
    get_solution(self) -> Solution
    # Calls state.get_solution()
```

```python
class L2DTransitionModel(TransitionModel):
    __init__(self, 
             affectations: numpy.array, 
             durations: numpy.array
             ) -> L2DTransitionModel 
    run(self, action: int) -> None
```

```python
class RewardModel(abc.ABC):
    __init__(self, affectations: numpy.array, durations: numpy.array) -> RewardModel
    @abstractmethod
    evaluate(self, state: State, action: int, next_state: State) -> int
```

```python
class L2DRewardModel(RewardModel):
    __init__(self, affectations: numpy.array, durations: numpy.array) -> RewardModel
    evaluate(self, state: State, action: int, next_state: State) -> int
```

## State

```python
class State:
    graph: networkX.Graph
    __init__(self, affectations: numpy.array, durations: numpy.array) -> State
    done(self) -> bool
    # Checks if current state is terminal or not
    reset(self) -> None
    # Reset the state to the initial state, according to affectations and durations
    to_torch_geometric(self, node_encoding: str) -> torch_geometric.data.Data
    # Compute an observation matching the gym.Env interface
    get_machine_availability(self, machine_id: int) -> List
    # Gives the times when the asked machine is available
    set_precedency(self, 
                   first_job_id: int, 
                   first_task_id: int, 
                   second_job_id: int, 
                   second_task_id: int
                   ) -> bool
    # Sets first task before second task if possible. Returns a boolean indicating
    # wheter the operation succeded or failed.
    get_solution(self) -> Solution
    # Returns the solution associated to the State, if self.done() is True. Else returns
    # False
    affect_node(self, node_id: int) -> None
```

## Utils

```python
def generate_problem(n_jobs: int, 
                     n_machines: int, 
                     max_duration: int
                     ) -> Tuple[numpy.array, numpy.array]
```
