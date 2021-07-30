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
    
    __init__(self, 
             env: Env, 
             n_epochs: int = None,
             gamma: float = None,
             clip_range: float = None,
             ent_coef: float = None,
             vf_coef: float = None,
             lr: float = None,
             model: PPO = None
             ) -> Agent
    save(self, path: str) -> None
    @classmethod
    load(cls, path: str) -> Agent
    # Loads an already trained agent, to allow using pretrained agents
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
    @abstractmethod
    get_mask(self) -> torch.Tensor
    # Computes the mask associated with internal state
    get_graph(self) -> torch_geometric.data.Data 
    # Calls state.to_torch_geometric()
    done(self) -> bool
    # Calls state.done()
    reset(self) -> None
    # Calls state.reset()
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
    __init__(self) -> RewardModel
    @abstractmethod
    evaluate(self, obs: Observation, action: int, next_obs: Observation) -> int
```

```python
class L2DRewardModel(RewardModel):
    __init__(self, affectations: numpy.array, durations: numpy.array) -> RewardModel
    evaluate(self, obs: Observation, action: int, next_obs: Observation) -> int
```

## State

```python
class State:
    graph: networkX.DiGraph
    task_completion_times: numpy.array
    is_affected: numpy.array
    __init__(self, affectations: numpy.array, durations: numpy.array) -> State
    reset(self) -> None
    # Reset the state to the initial state, according to affectations and durations
    done(self) -> bool
    # Checks if current state is terminal or not
    to_torch_geometric(self, node_encoding: str) -> torch_geometric.data.Data
    # Compute an observation matching the gym.Env interface
    set_precedency(self, 
                   first_node_id: int, 
                   second_node_id: int
                   ) -> bool
    # Sets first task before second task if possible. Returns a boolean indicating
    # wheter the operation succeded or failed.
    affect_node(self, node_id: int) -> None
    # Turns is_affected to 1 for the current node_id
    get_machine_occupancy(self, machine_id: int) -> List
    # Gives the times when the asked machine is available
    get_solution(self) -> Solution
    # Returns the solution associated to the State, if self.done() is True. Else returns
    # False
    get_first_unaffected_task(self, job_id: int) -> list
    # Returns the index of first unaffected task for the specified job
    get_job_availability(self, job_id: int, task_id: int) -> int
    # Returns the first time at which the current task is available to begin
```

## Utils

```python
def generate_problem(n_jobs: int, 
                     n_machines: int, 
                     max_duration: int
                     ) -> Tuple[numpy.array, numpy.array]

def node_to_job_and_task(node_id: int, n_machines: int) -> tuple[int, int]

def job_and_task_to_node(job_id: int, task_id: int, n_machines: int) -> int

def apply_mask(tensor: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int]

def solve_jssp(affectations: numpy.array, durations: numpy.array) -> Solution
# Solve a JSSP using OR-tools as a solver
```

```python
class Observation:
    n_nodes
    features
    edge_index
    mask
    @classmethod
    from_gym_observation(cls, gym_observation: dict) -> Observation
    @classmethod
    from_torch_geometric(cls, 
                         graph: torch_geometric.data.Data, 
                         mask: torch.Tensor
                         ) -> Observation
    get_batch_size(self) -> int
    get_n_nodes(self) -> int
    get_mask(self) -> torch.Tensor
    to_torch_geometric(self) -> torch_geometric.data.Data
    to_gym_observation(self) -> dict
    drop_node_ids(self) -> torch.Tensor
    # Since node_ids are added to the features (to avoid random shuffling), but we 
    # don't want to process them in graph embeddings networks, this function drops them
