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

# import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor

import torch

from ..utils.taillard_rcpsp import TaillardRcpsp
from .gstate import GState as State
from .observation import EnvObservation
from .reward_models.graph_terminal_reward_model import GraphTerminalRewardModel
from .transition_models.transition_model import TransitionModel


class GEnv:
    def __init__(
        self, problem_description, env_specification, pb_ids, validate=False, pyg=False
    ):
        self.problem_description = problem_description
        self.pyg = pyg
        self.env_specification = env_specification
        self.validate = validate
        self.transition_model_config = problem_description.transition_model_config
        self._create_transition_model()
        self.reward_model_config = problem_description.reward_model_config
        self._create_reward_model()
        self.observe_conflicts_as_cliques = (
            env_specification.observe_conflicts_as_cliques
        )
        self.observe_subgraph = env_specification.observe_subgraph
        self.deterministic = problem_description.deterministic
        self.n_features = self.env_specification.n_features
        self.observation_space = self.env_specification.observation_space
        self.action_space = self.env_specification.action_space
        self.pb_ids = pb_ids
        random.shuffle(self.pb_ids)
        self.pb_index = -1
        self.succ_cache = {}

        self.reset()

    def _problem_init(self):
        self.pb_index += 1
        if self.pb_index == len(self.pb_ids):
            random.shuffle(self.pb_ids)
            self.pb_index = 0

        if self.validate:
            self.problem = self.problem_description.test_psps[
                self.pb_ids[self.pb_index]
            ]
        else:
            self.problem = self.problem_description.train_psps[
                self.pb_ids[self.pb_index]
            ]

        if isinstance(self.problem, dict):
            self.n_jobs = self.problem["n_jobs"]
            self.n_modes = self.problem["n_modes"]
            self.n_resources = self.problem["n_resources"]
        else:
            self.n_jobs = self.problem.n_jobs
            self.n_modes = self.problem.n_modes
            self.n_resources = self.problem.n_resources

        # adjust n_jobs if we are going to sample or chunk
        self.sampled_jobs = None
        if self.env_specification.sample_n_jobs != -1:
            self.n_jobs = self.env_specification.sample_n_jobs
            self.problem, self.n_modes = self.sample_problem(self.problem, self.n_jobs)
        if self.env_specification.chunk_n_jobs != -1:
            self.n_jobs = self.env_specification.chunk_n_jobs
            self.problem, self.n_modes = self.chunk_problem(self.problem, self.n_jobs)
        if self.env_specification.random_taillard:
            # The problem instance can be something else than TaillardRcpsp
            # in case we are validating on non-random rcpsp instances. This
            # happens if we use `--random_taillard` and `--test_dir`.
            # in case of validation problems are already sampled
            if isinstance(self.problem, TaillardRcpsp) and not self.validate:
                self.problem = self.problem.sample()

        if self.problem_description.reward_model_config == "tardiness":
            if self.problem.due_dates is None:
                print("WARNING: tardiness reward requires but no due dates found")
            else:
                self.reward_model.set_tardiness()

    def close(self):
        pass

    def step(self, action):
        # Running the transition model on the current action
        node_id = self.action_to_node_id(action)
        n_actions = self.transition_model.run(self.state, node_id)

        # Getting next observation
        full_observation = self.observe()
        self.current_observation = self.process_obs(full_observation)

        # Getting the reward associated with the current action
        reward = self.reward_model.evaluate(self.state)

        self.sum_reward += reward

        # Getting final necessary information
        done = self.done()
        # gym_observation = next_obs.to_gym_observation()
        info = {
            "episode": {"r": self.sum_reward, "l": 1 + self.n_steps},
            "mask": self.action_masks(),
        }
        self.n_steps += n_actions

        return self.current_observation, reward, done, False, info

    def reset(self, soft=False):
        # Reset the internal state, but do not sample a new problem
        if soft:
            self.state.reset()

        # Reset the state by creating a new one
        # also may select a different problem
        else:
            self._problem_init()
            self._create_state()

        # Get the new observation
        full_observation = self.observe()
        self.current_observation = self.process_obs(full_observation)

        self.n_steps = 0
        self.sum_reward = 0
        info = {
            "episode": {"r": 0, "l": 0},
            "mask": self.action_masks(),
        }

        return self.current_observation, info

    # def succ_cached(self, graph, n):
    #     if n in self.succ_cache.keys():
    #         return self.succ_cache[n]
    #     # normal version
    #     # suc = set(graph.successors(n, etype="prec").tolist())
    #     # skip dgl inernal test
    #     suc = set(graph._graph.successors(graph.get_etype_id("prec"), n).tolist())
    #     self.succ_cache[n] = suc
    #     return suc

    def process_obs(self, full_observation):
        if not self.observe_subgraph:
            return full_observation
        frontier_nodes = torch.tensor(list(self.state.nodes_in_frontier))
        present_nodes = (torch.where(self.state.selectables() == 1)[0]).tolist()
        if self.env_specification.observation_horizon_step != 0:
            local_present_nodes = present_nodes
            future_nodes_step = set()
            for i in range(self.env_specification.observation_horizon_step):
                succs = set()
                for n in local_present_nodes:
                    # vanilla version
                    # outs = set(full_observation.successors(n, etype="prec").tolist())
                    # cached version
                    # outs = self.succ_cached(full_observation, n)
                    outs = set(full_observation.successors(n).tolist())
                    # nocheck version
                    # outs = set(
                    #     full_observation._graph.successors(
                    #         full_observation.get_etype_id("prec"), n
                    #     ).tolist()
                    # )
                    succs = succs.union(outs)
                local_present_nodes = succs
                future_nodes_step = future_nodes_step.union(local_present_nodes)
        else:
            future_nodes_step = set()

        if (
            self.env_specification.observation_horizon_time != 0
            and len(present_nodes) != 0
        ):
            present_date = torch.max(self.state.tct(present_nodes))
            earlier_start_times = (
                self.state.all_tct()[:, 1] - self.state.all_durations()[:, 1]
            )
            future_nodes_time = torch.where(
                torch.logical_and(
                    earlier_start_times > present_date,
                    earlier_start_times
                    < present_date + self.env_specification.observation_horizon_time,
                )
            )[0]
            future_nodes_time = set(future_nodes_time.tolist())
        else:
            future_nodes_time = set()

        future_nodes = future_nodes_step.union(future_nodes_time)

        if len(future_nodes) == 0:
            past_nodes = torch.where(self.state.get_pasts())[0]
            future_nodes = set(range(full_observation.num_nodes())) - set(
                past_nodes.tolist()
            )

        nodes_to_keep = torch.tensor(
            list(
                (
                    future_nodes.union(set(present_nodes)).union(
                        set(frontier_nodes.tolist())
                    )
                )
            )
        )

        # subgraph = dgl.node_subgraph(full_observation, nodes_to_keep)
        subgraph = full_observation.node_subgraph(nodes_to_keep)
        return subgraph

    def action_to_node_id(self, action):
        if not self.observe_subgraph:
            return action
        return self.current_observation.subid_to_origid(action)

    def process_mask(self, full_mask):
        if not self.observe_subgraph:
            return full_mask
        # return full_mask[self.current_observation.ndata[dgl.NID]]
        return self.current_observation.fullmask_to_submask(full_mask)

    def get_solution(self):
        sol = self.state.get_solution()
        if self.problem_description.reward_model_config == "makespan":
            sinks = torch.where(self.state.types() == 1)[0]
            sinks_makespans = self.state.tct(sinks)
            max_makespan = torch.max(sinks_makespans)
            sol._criterion = max_makespan
        elif self.problem_description.reward_model_config == "tardiness":
            with_due_dates = [
                i for i, v in enumerate(self.problem.due_dates) if v != None
            ]
            with_due_dates = torch.tensor(with_due_dates, dtype=torch.int64)
            due_dates = torch.tensor(
                [v for i, v in enumerate(self.problem.due_dates) if v != None],
                dtype=torch.int64,
            )
            wdd_tct = self.state.tct_real(with_due_dates)
            tardy = wdd_tct - due_dates
            sol._criterion = torch.sum(tardy).item()

        return sol

    def render_solution(self, schedule, scaling=1.0):
        return self.state.render_solution(schedule, scaling)

    def render_fail(self):
        return self.state.render_fail()

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
            pyg=self.pyg,
        )

    def _create_transition_model(self):
        self.transition_model = TransitionModel(self.env_specification)

    def _create_reward_model(self):
        self.reward_model = GraphTerminalRewardModel()

    def observe(self):
        return self.state.observe()

    def done(self):
        return self.state.done()

    def action_masks(self):
        full_mask = self.transition_model.get_mask(self.state)

        mask = self.process_mask(full_mask)

        pad = torch.zeros(
            (self.env_specification.max_n_modes - len(mask)),
            dtype=torch.bool,
            device=self.state.device,
        )
        return torch.cat([mask, pad]).to(torch.device("cpu"))

    def render_solution(self, schedule, scaling=1.0):
        return self.state.render_solution(schedule, scaling)
