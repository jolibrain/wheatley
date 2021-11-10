import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np
import sys
import traceback

from env.l2d_transition_model import L2DTransitionModel
from env.intrisic_reward_model import IntrisicRewardModel
from env.l2d_reward_model import L2DRewardModel
from env.meta_reward_model import MetaRewardModel
from env.sparse_reward_model import SparseRewardModel
from env.tassel_reward_model import TasselRewardModel
from env.uncertain_reward_model import UncertainRewardModel
from utils.env_observation import EnvObservation
from utils.utils import generate_problem, generate_problem_durations

from config import MAX_N_NODES, MAX_N_EDGES, MAX_N_MACHINES, MAX_N_JOBS


class Env(gym.Env):
    def __init__(
        self,
        problem_description,
        normalize_input=False,
        input_list=[],
        add_force_insert_boolean=False,
        slot_locking=False,
        full_force_insert=False,
        fixed_distrib=False,
    ):
        # both cases are possible (one for fake env, the other is nominal case)
        self.fixed_distrib = fixed_distrib or (problem_description.durations.shape[2] > 1)
        n_features = 2 + len(input_list)
        if "duration" in input_list and self.fixed_distrib:
            n_features += 3
        if "one_hot_job_id" in input_list:
            n_features += MAX_N_JOBS - 1
        if "one_hot_machine_id" in input_list:
            n_features += MAX_N_MACHINES - 1
        if "total_job_time" in input_list and fixed_distrib:
            n_features += 3
        if "total_machine_time" in input_list and fixed_distrib:
            n_features += 3
        if "job_completion_percentage" in input_list and fixed_distrib:
            n_features += 3
        if "machine_completion_percentage" in input_list and fixed_distrib:
            n_features += 3
        if "mwkr" in input_list and fixed_distrib:
            n_features += 3

        if self.fixed_distrib:
            n_features += 3  # completion times is of dim 4 instead of  1

        self.n_features = n_features
        self.n_jobs = problem_description.n_jobs
        self.n_machines = problem_description.n_machines
        self.max_duration = problem_description.max_duration
        self.normalize_input = normalize_input
        self.input_list = input_list
        self.add_force_insert_boolean = add_force_insert_boolean
        self.slot_locking = slot_locking
        self.full_force_insert = full_force_insert

        self.action_space = Discrete(2 * MAX_N_EDGES if self.add_force_insert_boolean else MAX_N_EDGES)
        self.observation_space = Dict(
            {
                "n_jobs": Discrete(MAX_N_JOBS + 1),
                "n_machines": Discrete(MAX_N_MACHINES + 1),
                "n_nodes": Discrete(MAX_N_NODES + 1),
                "n_edges": Discrete(MAX_N_EDGES + 1),
                "features": Box(
                    # high is max_duration*n_machines due to lower bound method of calculation
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

        self._create_reward_model(self.fixed_distrib)

        self.reset()

    def step(self, action):
        obs = EnvObservation.from_torch_geometric(
            self.n_jobs,
            self.n_machines,
            self.transition_model.get_graph(self.normalize_input, self.input_list),
            self.transition_model.get_mask(),
        )
        first_node_id, second_node_id, boolean = self._convert_action_to_node_ids(action)
        if self.full_force_insert:
            force_insert = True
            slot_lock = False
        elif self.add_force_insert_boolean:
            force_insert = boolean
            slot_lock = False
        elif self.slot_locking:
            force_insert = False
            slot_lock = boolean
        else:
            force_insert = False
            slot_lock = False
        self.transition_model.run(first_node_id, second_node_id, force_insert, slot_lock)
        next_obs = EnvObservation.from_torch_geometric(
            self.n_jobs,
            self.n_machines,
            self.transition_model.get_graph(self.normalize_input, self.input_list),
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
        boolean = True
        if self.add_force_insert_boolean or self.slot_locking:
            boolean = True if action >= MAX_N_EDGES else False
            action = action % MAX_N_EDGES
        first_node_id = action // MAX_N_NODES
        second_node_id = action % MAX_N_NODES
        return first_node_id, second_node_id, boolean

    def reset(self, force_regenerate_real_durations=True):
        if self.generate_random_problems:
            self.affectations, self.durations = generate_problem(self.n_jobs, self.n_machines, self.max_duration)
        if self.durations.shape[2] > 1:
            if force_regenerate_real_durations or self.durations[0, 0, 0] == -1:
                self.durations = generate_problem_durations(self.durations)

        self._create_transition_model()

        observation = EnvObservation.from_torch_geometric(
            self.n_jobs,
            self.n_machines,
            self.transition_model.get_graph(self.normalize_input, self.input_list),
            self.transition_model.get_mask(),
        )

        self.n_steps = 0

        return observation.to_gym_observation()

    def get_solution(self):
        return self.transition_model.state.get_solution()

    def render_solution(self, schedule, scaling=1.0):
        return self.transition_model.state.render_solution(schedule, scaling)

    def _create_transition_model(self):

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

    def _create_reward_model(self):
        if self.reward_model_config == "L2D":
            self.reward_model = L2DRewardModel()
        elif self.reward_model_config == "Sparse":
            self.reward_model = SparseRewardModel()
        elif self.reward_model_config == "Tassel":
            self.reward_model = TasselRewardModel(self.affectations, self.durations, self.normalize_input)
        elif self.reward_model_config == "Intrinsic":
            self.reward_model = IntrisicRewardModel(self.n_features * self.n_nodes)
        elif self.reward_model_config == "Intrinsic_and_L2D":
            self.reward_model = MetaRewardModel(
                [L2DRewardModel, IntrisicRewardModel],
                [{}, {"observation_input_size": self.n_features * self.n_nodes, "n_nodes": self.n_nodes}],
                [0, 1],
                n_timesteps=1500000,
            )
        else:
            self.reward_model = UncertainRewardModel(self.reward_model_config)
