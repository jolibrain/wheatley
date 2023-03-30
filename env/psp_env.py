#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#

import gymnasium as gym
from gymnasium.spaces import Discrete, Dict, Box
import numpy as np

from env.transition_models.psp_transition_model import PSPTransitionModel
from env.reward_models.sparse_reward_model import SparseRewardModel
from utils.psp_env_observation import PSPEnvObservation as EnvObservation
from utils.utils import get_n_features
from env.psp_state import PSPState as State


class PSPEnv(gym.Env):
    def __init__(self, problem_description, env_specification, i, validate=False):
        self.problem_description = problem_description
        self.env_specification = env_specification
        self.sum_reward = 0
        self.i = i
        if validate:
            self.problem = problem_description.test_psps[i]
        else:
            self.problem = problem_description.train_psps[i]

        self.transition_model_config = problem_description.transition_model_config
        self.reward_model_config = problem_description.reward_model_config
        self.n_jobs = self.problem["n_jobs"]
        self.n_modes = self.problem["n_modes"]
        # adjust n_jobs if we are going to sample or chunk
        if env_specification.sample_n_jobs != -1:
            self.n_jobs = env_specification.sample_n_jobs
            self.problem, self.n_modes = self.sample_problem(self.problem, self.n_jobs)
        if env_specification.chunk_n_jobs != -1:
            self.n_jobs = env_specification.chunk_n_jobs
            self.problem, self.n_modes = self.chunk_problem(self.problem, self.n_jobs)
        self.deterministic = problem_description.deterministic

        self.observe_conflicts_as_cliques = (
            env_specification.observe_conflicts_as_cliques
        )

        self.n_features = self.env_specification.n_features
        self.action_space = Discrete(
            self.env_specification.max_n_nodes
            * (2 if self.env_specification.add_boolean else 1)
        )

        if self.env_specification.max_edges_factor > 0:
            shape = (
                2,
                self.env_specification.max_edges_factor
                * self.env_specification.max_n_nodes,
            )
        else:
            shape = (2, self.env_specification.max_n_edges)

        self.observation_space = Dict(
            {
                "n_jobs": Discrete(self.problem_description.max_n_jobs + 1),
                "n_resources": Discrete(self.env_specification.max_n_resources + 1),
                "n_nodes": Discrete(self.env_specification.max_n_nodes + 1),
                "n_edges": Discrete(self.env_specification.max_n_edges + 1),
                "features": Box(
                    low=0,
                    high=1000,
                    shape=(self.env_specification.max_n_nodes, self.n_features),
                ),
                "edge_index": Box(
                    low=0,
                    high=self.env_specification.max_n_nodes,
                    shape=shape,
                    dtype=np.int64,
                ),
            }
        )

        if self.observe_conflicts_as_cliques:
            n_conflict_edges = (
                2
                * sum(range(self.env_specification.max_n_modes))
                * self.env_specification.max_n_resources
            )

            shape_conflict_edges = (2, n_conflict_edges)

            self.observation_space["n_conflict_edges"] = Discrete(n_conflict_edges)
            self.observation_space["conflicts_edges"] = Box(
                low=0,
                high=self.env_specification.max_n_nodes,
                shape=(2, n_conflict_edges),
                dtype=np.int64,
            )
            self.observation_space["conflicts_edges_resourceinfo"] = Box(
                low=np.array([0, 0, 0]),
                high=np.array(
                    [
                        self.env_specification.max_n_resources,
                        self.env_specification.max_resource_request,
                        self.env_specification.max_resource_availability,
                    ]
                ),
                shape=(2, n_conflict_edges, 3),
                dtype=np.int64,
            )

        self.transition_model = None
        self.reward_model = None

        self.n_steps = 0

        self._create_reward_model()

        self.reset()

    def step(self, action):
        # Getting current observation
        obs = self.observe()

        # Running the transition model on the current action
        node_id = action
        self.transition_model.run(self.state, node_id)

        # Getting next observation
        next_obs = self.observe()

        # Getting the reward associated with the current action
        reward = self.reward_model.evaluate(
            obs,
            action,
            next_obs,
        )

        self.sum_reward += reward

        # if needed, remove tct from obs (reward is computed on tct on obs ... )
        if self.env_specification.do_not_observe_updated_bounds:
            next_obs.features = next_obs.features.clone()
            next_obs.features[:, 2:6] = -1

        # Getting final necessary information
        done = self.done()
        gym_observation = next_obs.to_gym_observation()
        info = {
            "episode": {"r": self.sum_reward, "l": 1 + self.n_steps},
            "mask": self.action_masks(),
        }
        self.n_steps += 1

        return gym_observation, reward, done, False, info

    def reset(self, soft=False):
        # Reset the internal state, but do not sample a new problem
        if soft:
            self.state.reset()

        # Reset the state by creating a new one
        else:
            self._create_state()

        # Reset the transition model by creating a new one
        self._create_transition_model()

        # Get the new observation
        observation = self.observe()

        self.n_steps = 0
        self.sum_reward = 0
        info = {
            "episode": {"r": 0, "l": 0},
            "mask": self.action_masks(),
        }

        return observation.to_gym_observation(), info

    def get_solution(self):
        return self.state.get_solution()

    def render_solution(self, schedule, scaling=1.0):
        return self.state.render_solution(schedule, scaling)

    def chunk_problem(self, problem, n_jobs):
        # TODO
        return problem, problem["n_modes"]

    def sample_jobs(self, problem, n_jobs):
        # TODO
        return problem, problem["n_modes"]

    def _create_state(self):
        self.state = State(
            self.env_specification,
            self.problem_description,
            self.problem,
            self.deterministic,
            observe_conflicts_as_cliques=self.observe_conflicts_as_cliques,
        )

    def _create_transition_model(self):
        self.transition_model = PSPTransitionModel(self.env_specification)

    def _create_reward_model(self):
        self.reward_model = TerminalRewardModel()

    def observe(self):
        if self.observe_conflicts_as_cliques:
            (
                features,
                edge_index,
                conflicts_edges,
                conflicts_edges_machineid,
            ) = self.state.to_features_and_edge_index(
                self.env_specification.normalize_input,
                self.env_specification.input_list,
            )
            # remove real duration from obs (in state for computing makespan on the fly)
            if not self.env_specification.observe_real_duration_when_affect:

                features = self.state.get_features_wo_real_dur()
            return EnvObservation(
                self.n_jobs,
                self.n_modes,
                features,
                edge_index,
                conflicts_edges,
                conflicts_edges_machineid,
                self.env_specification.max_n_jobs,
                self.env_specification.max_n_resources,
                self.env_specification.max_edges_factor,
            )

        else:
            features, edge_index = self.state.to_features_and_edge_index(
                self.env_specification.normalize_input,
                self.env_specification.input_list,
            )
            # remove real duration from obs (in state for computing makespan on the fly)
            if not self.env_specification.observe_real_duration_when_affect:
                features = self.state.get_features_wo_real_dur()
            return EnvObservation(
                self.n_jobs,
                self.n_modes,
                features,
                edge_index,
                None,
                None,
                self.problem_description.max_n_jobs,
                self.problem_description.max_n_modes,
                self.problem_description.max_n_resources,
                self.env_specification.max_edges_factor,
            )

    def done(self):
        return self.state.done()

    def action_masks(self):
        mask = self.transition_model.get_mask(self.state)

        pad = np.full(
            (self.env_specification.max_n_modes - len(mask),),
            False,
            dtype=bool,
        )
        return np.concatenate([mask, pad])
