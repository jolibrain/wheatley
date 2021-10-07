import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np

from env.l2d_transition_model import L2DTransitionModel
from env.l2d_reward_model import L2DRewardModel
from env.sparse_reward_model import SparseRewardModel
from utils.env_observation import EnvObservation
from utils.utils import generate_problem

from config import MAX_N_NODES, MAX_N_EDGES, MAX_N_MACHINES, MAX_N_JOBS


class Env(gym.Env):
    def __init__(
        self,
        problem_description,
        normalize_input=False,
        add_machine_id=False,
        one_hot_machine_id=False,
        add_pdr_boolean=False,
        slot_locking=False,
    ):

        if not add_machine_id:
            n_features = 2
        else:
            if not one_hot_machine_id:
                n_features = 3
            else:
                n_features = 2 + MAX_N_MACHINES
        self.n_jobs = problem_description.n_jobs
        self.n_machines = problem_description.n_machines
        self.max_duration = problem_description.max_duration
        self.normalize_input = normalize_input
        self.add_machine_id = add_machine_id
        self.one_hot_machine_id = one_hot_machine_id
        self.add_pdr_boolean = add_pdr_boolean
        self.slot_locking = slot_locking

        self.action_space = Discrete(2 * MAX_N_EDGES if self.add_pdr_boolean else MAX_N_EDGES)
        self.observation_space = Dict(
            {
                "n_jobs": Discrete(MAX_N_JOBS + 1),
                "n_machines": Discrete(MAX_N_MACHINES + 1),
                "n_nodes": Discrete(MAX_N_NODES + 1),
                "n_edges": Discrete(MAX_N_EDGES + 1),
                "features": Box(
                    # high is 99*n_machines due to lower bound method of calculation
                    low=0,
                    high=self.max_duration * MAX_N_MACHINES,
                    shape=(MAX_N_NODES, n_features),
                ),
                "edge_index": Box(
                    low=0,
                    high=MAX_N_NODES,
                    shape=(2, MAX_N_EDGES),
                    dtype=np.int64,
                ),
                "mask": Box(low=0, high=1, shape=(MAX_N_EDGES,)),
            }
        )
        self.n_nodes = self.n_machines * self.n_jobs

        self.affectations = problem_description.affectations
        self.durations = problem_description.durations
        self.transition_model_config = problem_description.transition_model_config
        self.reward_model_config = problem_description.reward_model_config

        self.generate_random_problems = False
        if problem_description.affectations is None and problem_description.durations is None:
            self.generate_random_problems = True

        self.transition_model = None
        self.reward_model = None

        self.n_steps = 0

        self.reset()

    def step(self, action):
        obs = EnvObservation.from_torch_geometric(
            self.n_jobs,
            self.n_machines,
            self.transition_model.get_graph(self.add_machine_id, self.normalize_input, self.one_hot_machine_id),
            self.transition_model.get_mask(),
        )
        first_node_id, second_node_id, boolean = self._convert_action_to_node_ids(action)
        if self.add_pdr_boolean:
            pdr_boolean = boolean
            slot_lock = False
        elif self.slot_locking:
            pdr_boolean = True
            slot_lock = boolean
        else:
            pdr_boolean = True
            slot_lock = False
        self.transition_model.run(first_node_id, second_node_id, pdr_boolean, slot_lock)
        next_obs = EnvObservation.from_torch_geometric(
            self.n_jobs,
            self.n_machines,
            self.transition_model.get_graph(self.add_machine_id, self.normalize_input, self.one_hot_machine_id),
            self.transition_model.get_mask(),
        )
        reward = self.reward_model.evaluate(
            obs,
            action,
            next_obs,
        )
        done = self.transition_model.done()
        gym_observation = next_obs.to_gym_observation()

        info = {"episode": {"r": reward, "l": 1 + self.n_steps * 2}}
        self.n_steps += 1

        return gym_observation, reward, done, info

    def _convert_action_to_node_ids(self, action):
        pdr_boolean = True
        if self.add_pdr_boolean or self.slot_locking:
            pdr_boolean = True if action >= MAX_N_EDGES else False
            action = action % MAX_N_EDGES
        first_node_id = action // MAX_N_NODES
        second_node_id = action % MAX_N_NODES
        return first_node_id, second_node_id, pdr_boolean

    def reset(self):
        if self.generate_random_problems:
            self.affectations, self.durations = generate_problem(self.n_jobs, self.n_machines, self.max_duration)
        self._create_transition_and_reward_model()
        observation = EnvObservation.from_torch_geometric(
            self.n_jobs,
            self.n_machines,
            self.transition_model.get_graph(self.add_machine_id, self.normalize_input, self.one_hot_machine_id),
            self.transition_model.get_mask(),
        )

        self.n_steps = 0

        return observation.to_gym_observation()

    def get_solution(self):
        return self.transition_model.state.get_solution()

    def _create_transition_and_reward_model(self):

        if self.transition_model_config == "L2D":
            self.transition_model = L2DTransitionModel(
                self.affectations, self.durations, node_encoding="L2D", slot_locking=self.slot_locking
            )
        elif self.transition_model_config == "DenseL2D":
            self.transition_model = L2DTransitionModel(
                self.affectations, self.durations, node_encoding="DenseL2D", slot_locking=self.slot_locking
            )
        else:
            raise Exception("Transition model not recognized")

        if self.reward_model_config == "L2D":
            self.reward_model = L2DRewardModel()
        elif self.reward_model_config == "Sparse":
            self.reward_model = SparseRewardModel()
        else:
            raise Exception("Reward model not recognized")
