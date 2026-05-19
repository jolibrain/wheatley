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
from concurrent.futures import ThreadPoolExecutor

import torch
import random
from muutils.misc import stable_hash
from .state import State
from .reward_models.reward_model import ShortestPathRewardModel
from .transition_models.transition_model import MazeTransitionModel as TransitionModel
from .transition_models.transition_model_nonchrono import (
    MazeTransitionModel as TransitionModelNC,
)
from pp.domains.maze.generators.maze_generator import generate_mazes
from pp.graph.graph_factory import GraphFactory
import numpy as np
from pp.solution import Solution
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
from maze_dataset.maze.lattice_maze import NEIGHBORS_MASK

import sys


class Env:
    def __init__(
        self,
        problem_description,
        env_specification,
        pb_ids,
        validate=False,
        reset=True,
        lappe=None,
        rwpe=None,
        create_new_pbs=True,
        seed=0,
        walls=True,
        nonchrono=None,
        pp_agent_types=None,
        pp_allow_stay=False,
        pp_all_agents_must_finish=False,
        pp_maze_gen="dfs",
        pp_force_common_start=False,
        pp_num_goals=1,
        pp_random_num_goals=True,
        pp_danger_max_size=0,
        pp_danger_max_num=0,
        pp_danger_prob=1.0,
        pp_danger_multiplier=2.0,
        pp_goal_reward=0.0,
        **kwargs,
    ):
        self.problem_description = problem_description
        self.env_specification = env_specification
        self.nonchrono = nonchrono
        self.partial_sol = True
        if pp_agent_types is None:
            pp_agent_types = [0, 1]
        self.pp_agent_types = list(pp_agent_types)
        self.pp_force_common_start = bool(pp_force_common_start)
        self.pp_num_goals = max(1, int(pp_num_goals))
        self.pp_random_num_goals = bool(pp_random_num_goals)
        self.pp_danger_max_size = max(0, int(pp_danger_max_size))
        self.pp_danger_max_num = max(0, int(pp_danger_max_num))
        self.pp_danger_prob = float(max(0.0, min(1.0, pp_danger_prob)))
        self.pp_danger_multiplier = float(max(0.0, pp_danger_multiplier))
        self.pp_goal_reward = float(pp_goal_reward)
        # self.n_breaks = [1 if agent_type == 1 else 0 for agent_type in self.pp_agent_types]
        self.allow_stay = bool(pp_allow_stay) and len(self.pp_agent_types) > 1
        self.all_agents_must_finish = bool(pp_all_agents_must_finish)
        self._create_transition_model()
        self._create_reward_model()
        self.pb_index = 0
        self.pb_ids = pb_ids
        self.validate = validate
        self.lappe = lappe
        self.rwpe = rwpe
        self.create_new_pbs = create_new_pbs
        self.walls = walls
        self.pp_maze_gen = pp_maze_gen

        np.random.seed(seed)
        random.seed(seed)
        random.shuffle(self.pb_ids)
        if reset:
            self.reset()

    def reset(self, soft=False):
        # Reset the internal state, but do not sample a new problem
        if soft:  # XXX: with fixed_validation
            self._create_env()
            self._create_state(self.pp_agent_types)

        # Reset the state by creating a new one
        # also may select a different problem
        else:
            self._problem_init()
            self._create_env()
            self._create_state(self.pp_agent_types)

        # Get the new observation
        for agent_id in range(len(self.states)):
            self.observe(agent_id)
        obs = self.process_obs(self.graph)

        self.n_steps = 0
        self.sum_reward = 0
        info = {
            "episode": {"r": 0, "l": 0},
            "mask": self._all_action_masks(),
        }

        return obs, info

    def _problem_init(self):
        self.pb_index += 1
        if self.pb_index == len(self.pb_ids):
            random.shuffle(self.pb_ids)
            self.pb_index = 0
        if self.validate:
            self.problem = self.problem_description.test_pbs[self.pb_ids[self.pb_index]]
        else:
            if self.create_new_pbs:
                self.problem = generate_mazes(
                    self.problem_description.train_pbs[0].x,
                    1,
                    maze_gen=self.pp_maze_gen,
                    forbid_trivial=False,
                )[0][0]
            else:
                self.problem = self.problem_description.train_pbs[
                    self.pb_ids[self.pb_index]
                ]

    ##TODO: add agent type (int)
    def _create_state(self, agent_types):
        self.states = []
        for i, agent_type in enumerate(agent_types):
            n_breaks = 1 if agent_type == 1 else 0
            state = State(
                self,
                self._agent_start_nids[i],
                self._goal_nids,
                self.env_specification.max_n_steps,
                self.problem.ncells,
                nonchrono=self.nonchrono,
                n_breaks=n_breaks,
            )
            self.states.append(state)
        self._sync_goal_status_to_states()
        self._update_goal_features()
        self._update_danger_features()

    def _create_env(self):
        if hasattr(self, "problem") and self.problem is not None:
            reset_fn = getattr(self.problem, "reset_dynamic_changes", None)
            if callable(reset_fn):
                reset_fn()
        self._reset_caches()
        self._assign_agent_starts()
        self._assign_goals()
        self._assign_danger_zones()
        self._reset_graph()
        self._reset_coord()
        self.undoable_criterion = 100000
        self._reset_optimal()

    def _reset_caches(self):
        nodes = self.problem.nodes_coord()
        self._coord_to_nid = {}
        self._nid_to_coord = {}
        for n, c in enumerate(nodes):
            self._coord_to_nid[tuple(c)] = n
            self._nid_to_coord[n] = tuple(c)

    def _assign_agent_starts(self):
        starts = self.problem.agent_starts(
            len(self.pp_agent_types),
            force_common_start=self.pp_force_common_start,
        )
        self._agent_start_coords = [tuple(c) for c in starts]
        self._agent_start_nids = [
            self.coord_to_nid(c) for c in self._agent_start_coords
        ]

    def _sample_num_goals(self):
        """Sample the number of goals for the current maze."""
        if self.validate or not self.pp_random_num_goals:
            return self.pp_num_goals

        try:
            # Use a deterministic seed so the goal count is consistent for a given maze
            seed_val = stable_hash(str(self.problem.data_hash()))
        except Exception:
            seed_val = None

        if seed_val is not None:
            rng = random.Random(seed_val)
            return rng.randint(1, self.pp_num_goals)

        return random.randint(1, self.pp_num_goals)

    def _assign_goals(self):
        num_goals = self._sample_num_goals()
        goal_coords = self.problem.shared_goals(
            num_goals,
            exclude_coords=self._agent_start_coords,
        )
        self._goal_coords = [tuple(c) for c in goal_coords]
        self._goal_nids = [self.coord_to_nid(c) for c in self._goal_coords]
        self._goal_nid_to_idx = {nid: idx for idx, nid in enumerate(self._goal_nids)}
        self.goal_available = torch.ones(len(self._goal_nids), dtype=torch.bool)
        self.goal = self._goal_nids[0] if self._goal_nids else None
        self._current_num_goals = len(self._goal_nids)

    def _sync_goal_status_to_states(self):
        if not hasattr(self, "states"):
            return
        for state in self.states:
            state.set_goal_status(self.goal_available)

    def _assign_danger_zones(self):
        zones = self.problem.danger_zones(
            self.pp_danger_max_num,
            self.pp_danger_max_size,
        )
        self._danger_zone_coords = [list(zone) for zone in zones]
        self._danger_zone_mask = torch.zeros(self.problem.ncells, dtype=torch.bool)
        self._danger_zone_nids = set()
        for zone in self._danger_zone_coords:
            for coord in zone:
                nid = self.coord_to_nid(coord)
                self._danger_zone_mask[nid] = True
                self._danger_zone_nids.add(nid)

    def _reset_graph(self):
        edge_index = {}
        access_native = self.problem.adj_list()
        access = [[], []]
        access_cache = np.zeros((self.problem.ncells, self.problem.ncells), dtype=int)
        for n1 in range(self.problem.ncells):
            for n2 in range(self.problem.ncells):
                if n1 == n2:
                    continue
                if self._neigh(n1, n2):
                    access_cache[n1, n2] = 2  # wall

        for a in access_native:
            n1 = self.coord_to_nid(a[0])
            n2 = self.coord_to_nid(a[1])
            access[0].extend([n1, n2])
            access[1].extend([n2, n1])
            access_cache[n1, n2] = 1  # no wall
            access_cache[n2, n1] = 1
        edge_index["free"] = torch.tensor(access)
        if self.walls:
            edge_index["wall"] = torch.tensor(np.array(np.nonzero(access_cache == 2)))
        if self.partial_sol:
            # edge_index["neigh_graph"] = None
            # edge_index["neigh_graph_inv"] = None
            edge_index["path_graph"] = None
            edge_index["path_graph_inv"] = None

        self.graph = GraphFactory.create_graph(
            edge_index,
            self.problem.ncells,
        )

        deg = []
        for n in range(self.problem.ncells):
            preds = edge_index["free"][0][torch.where(edge_index["free"][1] == n)[0]]
            deg.append(len(preds))
        deg = torch.tensor(deg)
        self.graph.set_ndata("degree", deg)

        if self.lappe is not None:
            subg = self.graph._graph.edge_type_subgraph([("n", "free", "n")])
            addLapPe = AddLaplacianEigenvectorPE(self.lappe)
            subgh = subg.to_homogeneous()
            subgh = addLapPe(subgh)
            self.graph._graph[
                "n"
            ].laplacian_eigenvector_pe = subgh.laplacian_eigenvector_pe

        if self.rwpe is not None:
            subg = self.graph._graph.edge_type_subgraph([("n", "free", "n")])
            addRWPE = AddRandomWalkPE(self.rwpe)
            subgh = subg.to_homogeneous()
            subgh = addRWPE(subgh)
            self.graph._graph["n"].random_walk_pe = subgh.random_walk_pe

    def _neigh(self, n1, n2):
        c1 = self.nid_to_coord(n1)
        c2 = self.nid_to_coord(n2)
        if (abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])) == 1:
            return True
        return False

    def _reset_coord(self):
        coords = []
        for i in range(self.problem.ncells):
            coords.append(self.nid_to_coord(i))
        self.graph.set_ndata("coord", torch.tensor(coords))
        self.graph.set_ndata("norm_coord", torch.tensor(coords) / self.problem.x)

        start = torch.zeros(self.problem.ncells)
        start[self._agent_start_nids[0]] = 1
        self.graph.set_ndata("start", start)
        for agent_id, start_nid in enumerate(self._agent_start_nids):
            agent_start = torch.zeros(self.problem.ncells)
            agent_start[start_nid] = 1
            self.graph.set_ndata(f"agent_{agent_id}_start", agent_start)

        self._update_goal_features()

    def _reset_optimal(self):
        opt = self.problem.optimal
        self.optimal = []
        for c in opt:
            self.optimal.append(self.coord_to_nid(tuple(c)))
            self.optimal_native = opt

    def coord_to_nid(self, c):
        return self._coord_to_nid[tuple(c)]

    def nid_to_coord(self, nid):
        return self._nid_to_coord[nid]

    def pos_neigh_nodes(self, pos):
        pos_coord = self.nid_to_coord(pos)
        coord_neigh = self.problem.neigh_of(pos_coord)
        return [self.coord_to_nid(c) for c in coord_neigh]

    def _get_coord_neighbors_all(self, c):
        c = np.array(c)
        neighbors = [
            neighbor
            for neighbor in (c + NEIGHBORS_MASK)
            if 0 <= neighbor[0] < self.problem.solvedMaze.grid_shape[0]
            and 0 <= neighbor[1] < self.problem.solvedMaze.grid_shape[1]
        ]
        return np.array(neighbors)

    def all_pos_neigh_nodes(self, pos):
        pos_coord = self.nid_to_coord(pos)
        all_coord_neigh = self._get_coord_neighbors_all(pos_coord)
        return [self.coord_to_nid(c) for c in all_coord_neigh]

    def _update_goal_features(self):
        goal = torch.zeros(self.problem.ncells)
        for idx, nid in enumerate(self._goal_nids):
            if self.goal_available[idx]:
                goal[nid] = 1
        self.graph.set_ndata("goal", goal)
        goal_flag = self.goal_available.to(dtype=torch.float32)
        self.graph.set_global_data("goal_available", goal_flag)
        if hasattr(self, "states"):
            for agent_id in range(len(self.states)):
                self.graph.set_global_data(
                    f"agent_{agent_id}_goal_available",
                    goal_flag,
                )

    def _update_danger_features(self):
        if not hasattr(self, "_danger_zone_mask"):
            self._danger_zone_mask = torch.zeros(self.problem.ncells, dtype=torch.bool)
        danger = self._danger_zone_mask.to(dtype=torch.float32)
        self.graph.set_ndata("danger", danger)

    def _danger_cost_multiplier(self, nid):
        if (
            self.pp_danger_prob <= 0.0
            or not hasattr(self, "_danger_zone_mask")
            or self._danger_zone_mask.numel() == 0
        ):
            return 1.0
        idx = int(nid)
        if idx < 0 or idx >= self._danger_zone_mask.numel():
            return 1.0
        if not bool(self._danger_zone_mask[idx]):
            return 1.0
        if np.random.random() < self.pp_danger_prob:
            return self.pp_danger_multiplier  # danger multiplier
        return 1.0

    def is_in_danger(self, nid):
        if not hasattr(self, "_danger_zone_mask"):
            return False
        idx = int(nid)
        if idx < 0 or idx >= self._danger_zone_mask.numel():
            return False
        return bool(self._danger_zone_mask[idx])

    def all_goals_consumed(self):
        if not hasattr(self, "goal_available"):
            return False
        if self.goal_available.numel() == 0:
            return False
        return not self.goal_available.any().item()

    def try_consume_goal(self, nid):
        if torch.is_tensor(nid):
            scalar_nid = int(nid.item())
        else:
            scalar_nid = int(nid)
        idx = self._goal_nid_to_idx.get(scalar_nid, None)
        if idx is None:
            return None
        if not self.goal_available[idx]:
            return None
        self.goal_available[idx] = False
        self._update_goal_features()
        self._sync_goal_status_to_states()
        return idx

    def close(self):
        pass

    def step(self, action, agent_id=0):
        if self.nonchrono is None:
            # remove dim of action, needed only for nonchrono_path
            action = action.squeeze(0)
            return self.step_chrono(action, agent_id)
        return self.step_nonchrono(action, agent_id)

    def step_nonchrono(self, action, agent_id=0):
        if action.shape[-1] == 1:  # single agent case
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).squeeze(-1)
            else:
                action = action.squeeze(-1)

            obs, reward, done, terminated, info = self._step_single_agent(
                agent_id, action
            )
            self.sum_reward += reward
            self.n_steps += 1
            info["episode"] = {"r": self.sum_reward, "l": self.n_steps}

            return obs, reward, done, terminated, info
        else:
            if isinstance(action, (list, tuple)):
                return self.step_agents_nonchrono(list(action))
            return self.step_agents_nonchrono(action.tolist())

    def step_chrono(self, action, agent_id=0):
        # print('env step')

        # print('action=', action)

        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                action = int(action.item())
            else:
                # print('env step agents') #XXX: where rollouts go
                return self.step_agents(action.tolist())
        if torch.is_tensor(action):
            if action.ndim == 0:
                action = int(action.item())
            else:
                # print('env step agents2')
                return self.step_agents(action.tolist())
        if isinstance(action, (list, tuple)):
            # print('env step agents3') #XXX: where agent_validator predict goes, and random_agent
            return self.step_agents(list(action))

        # print('step single agent')
        obs, reward, done, terminated, info = self._step_single_agent(agent_id, action)
        self.sum_reward += reward
        self.n_steps += 1
        info["episode"] = {"r": self.sum_reward, "l": self.n_steps}
        # if done:
        #     print(
        #         "env final steps=",
        #         self.n_steps,
        #         " / done=",
        #         done,
        #         " / terminated=",
        #         terminated,
        #         " / reward=",
        #         self.sum_reward,
        #     )

        return obs, reward, done, terminated, info

    def step_agents_nonchrono(self, actions):
        raise NotImplementedError

    def step_agents(self, actions):
        if len(actions) != len(self.states):
            raise ValueError(
                f"Received {len(actions)} actions for {len(self.states)} agents"
            )

        shared_reward = 0.0
        done_flags = []
        terminated_flags = []
        last_obs = None

        for agent_id, action in enumerate(actions):
            if self.all_goals_consumed():
                done_flags.append(True)
                terminated_flags.append(True)
                continue
            obs, reward, done, terminated, info = self._step_single_agent(
                agent_id, action
            )
            last_obs = obs

            shared_reward += reward

            # print('agent_id=', agent_id, 'reward=', reward, 'steps=', self.n_steps)

            done_flags.append(done)
            terminated_flags.append(terminated)

        # print('shared_reward=', shared_reward)

        self.sum_reward += shared_reward
        self.n_steps += 1

        if self.all_agents_must_finish:
            done_result = all(done_flags) if done_flags else False
            terminated_result = all(terminated_flags) if terminated_flags else False
        else:
            done_result = any(done_flags)
            terminated_result = any(terminated_flags)

        info = {
            "mask": self._all_action_masks(),
            "episode": {"r": self.sum_reward, "l": self.n_steps},
        }
        # if done_result:
        #     print(
        #         "env final steps=",
        #         self.n_steps,
        #         " / done=",
        #         done_result,
        #         " / terminated=",
        #         terminated_result,
        #         " / reward=",
        #         self.sum_reward,
        #     )

        if last_obs is None:
            last_obs = self.process_obs(self.graph)

        return last_obs, shared_reward, done_result, terminated_result, info

    def _step_single_agent(self, agent_id, action):
        # Getting the reward associated with the current action
        # get origin node from self.state, to get edge
        # only needed if world has been modified by agent action
        if self.nonchrono is None:
            curr_pos = self.states[agent_id].pos
            danger_multiplier = self._danger_cost_multiplier(curr_pos)
            self.states[agent_id].step_cost_multiplier = danger_multiplier
            eid = self.graph.find_edge("wall", curr_pos, action)
        else:
            eid = None

        prev_state_data_for_reward = self.reward_model.get_data_for_reward(
            self.states[agent_id]
        )

        # apply transition model
        self.transition_model.run(self.states[agent_id], action)

        # remove a wall edge if it was destroyed by agent action
        if eid is not None:
            self.graph.remove_edge(eid, "wall")
            eid_back = self.graph.find_edge(
                "wall", action, curr_pos
            )  # here since removal modifies edge id
            self.graph.remove_edge(eid_back, "wall")

            # call add_edges to put the free edge
            # print('added free edge between:', self.nid_to_coord(curr_pos), self.nid_to_coord(action))
            self.graph.add_edge(curr_pos, action, "free")
            self.graph.add_edge(action, curr_pos, "free")

            # update degree

            # keep maze-dataset representation in sync for other agents
            self.problem.break_wall(
                self.nid_to_coord(curr_pos),
                self.nid_to_coord(action),
            )

        obs = self.process_obs(self.observe(agent_id))
        reward = self.reward_model.evaluate(
            self.states[agent_id], self.sum_reward, prev_state_data_for_reward
        )
        if self.nonchrono is None:
            self.states[agent_id].step_cost_multiplier = 1.0

        # Getting final necessary information
        done = self.done(agent_id)
        info = {
            "mask": self._all_action_masks(),
        }

        return obs, reward, done, False, info

    def process_obs(self, full_observation):
        return full_observation

    def process_mask(self, full_mask):
        return full_mask

    def get_solution(self, agent_id=0):
        if self.states[agent_id].path is None:
            sol_native = []
        else:
            sol_native = [
                self._agent_start_coords[agent_id],
            ] + [self.nid_to_coord(c) for c in self.states[agent_id].path]
        return Solution(
            self.states[agent_id].path, self.reward_model, self.optimal, sol_native
        )

    def _create_transition_model(self):
        if self.nonchrono is not None:
            self.transition_model = TransitionModelNC(self, self.nonchrono)
        else:
            self.transition_model = TransitionModel(self)

    def _create_reward_model(self):
        self.reward_model = ShortestPathRewardModel(
            self.env_specification,
            goal_reward=self.pp_goal_reward,
            nonchrono=self.nonchrono,
        )

    def observe(self, agent_id=0):
        agent_prefix = f"agent_{agent_id}_"
        if self.nonchrono is not None:
            self.graph.set_ndata(
                f"{agent_prefix}selected", self.states[agent_id].selected
            )

            if self.nonchrono == "path" and self.partial_sol:
                if len(self.states[agent_id].path_graph) != 0:
                    edges = torch.tensor(self.states[agent_id].path_graph)
                    sources = edges[:, 0]
                    dests = edges[:, 1]
                    self.graph.set_edges(sources, dests, "path_graph")
                    self.graph.set_edges(dests, sources, "path_graph_inv")
                else:
                    self.graph.clear_edges("path_graph")
                    self.graph.clear_edges("path_graph_inv")

                self.graph.set_ndata("in_path", self.states[agent_id].node_in_path)

        else:
            self.graph.set_global_data(
                f"{agent_prefix}pos", self.states[agent_id].pos
            )  ##XXX: unused
            self.graph.set_ndata(
                f"{agent_prefix}cur_pos", self.states[agent_id].cur_pos
            )
            self.graph.set_ndata(
                f"{agent_prefix}visited", self.states[agent_id].visited
            )

        # make n_breaks a tensor of ncells with repeated value of self.state.n_breaks
        n_breaks_t = torch.tensor(
            [self.states[agent_id].n_breaks] * self.problem.ncells, dtype=torch.int64
        )

        self.graph.set_ndata(f"{agent_prefix}n_breaks", n_breaks_t)

        return self.graph

    def done(self, agent_id=0):
        return self.states[agent_id].done()

    def action_masks(self, agent_id=0):
        full_mask = self.transition_model.get_mask(self.states[agent_id])

        mask = self.process_mask(full_mask)

        return mask

    def _all_action_masks(self):
        masks = [self.action_masks(agent_id) for agent_id in range(len(self.states))]
        return masks
