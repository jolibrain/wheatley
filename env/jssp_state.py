#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#    Antoine Jacquet <antoine.jacquet@jolibrain.com>
#    Michel Thomazo <thomazo.michel@gmail.com>
#    Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
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

from copy import deepcopy
from queue import PriorityQueue

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch

import datetime
import pandas as pd
import plotly.figure_factory as ff
import cv2

from problem.solution import Solution
from utils.utils import (
    node_to_job_and_task,
    job_and_task_to_node,
    get_n_features,
    compute_conflicts_cliques,
)


class JSSPState:
    def __init__(
        self,
        affectations,
        durations,
        max_n_jobs,
        max_n_machines,
        deterministic=True,
        feature_list=[],
        observe_conflicts_as_cliques=False,
    ):
        self.affectations = affectations
        self.original_durations = durations.copy()
        self.n_jobs = self.affectations.shape[0]
        self.n_machines = self.affectations.shape[1]
        self.n_nodes = self.n_jobs * self.n_machines
        self.observe_conflicts_as_cliques = observe_conflicts_as_cliques

        self.max_n_jobs = max_n_jobs
        self.max_n_machines = max_n_machines

        # no more one hot due to perf issues when adding conflcits as cliques
        self.not_one_hot_machine_id = np.zeros((max_n_machines, max_n_machines))
        for i in range(max_n_machines):
            self.not_one_hot_machine_id[i][:] = i
        self.one_hot_machine_id = np.zeros((max_n_machines, max_n_machines))
        for i in range(max_n_machines):
            self.one_hot_machine_id[i][i] = 1

        self.deterministic = deterministic

        self.colors = self.generate_colors()

        self.graph = None

        # cache
        self.same_job = {}

        self.init_features_offset(feature_list)
        self.features = torch.zeros(
            (
                self.n_nodes,
                get_n_features(feature_list, self.max_n_jobs, self.max_n_machines),
            )
        )

        self.affected = None
        self.is_observed = None
        self.durations = None
        self.n_jobs_per_machine = None
        self.n_machines_per_job = None

        # Used to compute the features
        self.max_duration = None
        self.max_completion_time = None
        self.total_job_time = None
        self.total_machine_time = None
        self.total_machine_time_job_task = None
        self.job_completion_time = None
        self.machine_completion_time = None

        self.set_not_one_hot_machine_id()
        if self.observe_conflicts_as_cliques:
            self.compute_conflicts_cliques()
            self.set_one_hot_machine_id()
        self.n_jobs_per_machine = np.array(
            [(self.affectations == m).sum() for m in range(self.n_machines)]
        )
        self.n_machines_per_job = np.array(
            [
                self.n_machines - (self.affectations[j] == -1).sum()
                for j in range(self.n_jobs)
            ]
        )
        self.max_duration = np.max(self.original_durations.flatten())
        self.edges = None

        self.reset()

    def cache_last_task_on_machine(self, machine_id, task):
        self.last_task_on_machine[machine_id] = task

    def reset_is_affected(self):
        self.affected = np.zeros_like(self.affectations)
        self.features[:, 0] = 0

    def is_affected(self, job_id, task_id):
        return self.features[
            job_and_task_to_node(job_id, task_id, self.max_n_machines), 0
        ].item()

    def is_affected_by_node(self, node_id):
        return self.features[node_id, 0].item()

    def affect(self, node_id):
        self.features[node_id, 0] = 1
        self.affected[node_to_job_and_task(node_id, self.max_n_machines)] = 1

    def set_all_task_completion_times(self, tct):
        self.features[:, 1:5] = torch.as_tensor(tct).reshape(
            (self.n_jobs * self.n_machines, -1)
        )

    def reset_task_completion_times(self):
        tct = np.cumsum(
            np.where(self.original_durations < 0, 0, self.original_durations), axis=1
        )
        if not self.deterministic:
            tct[:, :, 0] = -1
        else:
            tct[:, :, 0] = np.where(self.durations[:, :, 0] == -1, -1, tct[:, :, 0])
        self.features[:, 1:5] = (
            torch.as_tensor(tct).clone().reshape((self.n_jobs * self.n_machines, -1))
        )

    def get_task_completion_times(self, node_id):
        return self.features[node_id, 1:5]

    def get_all_task_completion_times(self):
        return self.features[:, 1:5]

    def set_task_completion_times(self, node_id, ct):
        j, t = node_to_job_and_task(node_id, self.max_n_machines)
        self.features[node_id, 1:5] = ct.clone()

    # no more one hot due to perf issues when adding conflcits as cliques
    def set_not_one_hot_machine_id(self):
        self.features[:, 6 : 6 + self.max_n_machines] = (
            torch.zeros(self.max_n_machines) - 1
        )
        for job_id in range(self.n_jobs):
            for task_id in range(self.n_machines):
                machine_id = self.affectations[job_id, task_id]
                node_id = job_and_task_to_node(job_id, task_id, self.max_n_machines)
                if machine_id != -1:
                    self.features[
                        node_id, 6 : 6 + self.max_n_machines
                    ] = torch.as_tensor(self.not_one_hot_machine_id[machine_id])

    def set_one_hot_machine_id(self):
        for job_id in range(self.n_jobs):
            for task_id in range(self.n_machines):
                machine_id = self.affectations[job_id, task_id]
                node_id = job_and_task_to_node(job_id, task_id, self.max_n_machines)
                if machine_id == -1:
                    self.features[node_id, 6 : 6 + self.max_n_machines] = torch.zeros(
                        self.max_n_machines
                    )
                else:
                    self.features[
                        node_id, 6 : 6 + self.max_n_machines
                    ] = torch.as_tensor(self.one_hot_machine_id[machine_id])

    def reset_durations(self):
        self.durations = self.original_durations.copy()
        self.durations[:, :, 0] = -1
        if "duration" in self.features_offset:
            dof = self.features_offset["duration"]
            self.features[:, dof[0] : dof[1]] = (
                torch.as_tensor(self.durations).reshape(-1, 4).clone()
            )

    def get_durations(self, node_id):
        if "duration" in self.features_offset:
            dof = self.features_offset["duration"]
            # durs = self.features[node_id, dof[0] : dof[1]]
            # return durs.clone()
            return self.features[node_id, dof[0] : dof[1]]
        return torch.as_tensor(
            self.durations[node_to_job_and_task(node_id, self.max_n_machines)],
            dtype=torch.float,
        )

    def reset(self):
        self.last_task_on_machine = {}

        self.edges = [
            (
                job_index * self.n_machines + i,
                job_index * self.n_machines + i + 1,
            )
            for i in range(self.n_machines - 1)
            for job_index in range(self.n_jobs)
        ]

        self.graph = nx.DiGraph(self.edges)
        self.numpy_edges = np.array(self.edges)
        self.edges = []

        self.reset_durations()
        self.reset_task_completion_times()
        self.reset_is_affected()

        self.is_observed = np.zeros_like(self.affectations)
        self.max_completion_time = torch.max(self.features[:, 1:5].flatten()).item()

        self.compute_pre_features()

    def compute_conflicts_cliques(self):
        (
            self.conflicts_edges,
            self.conflicts_edges_machineid,
        ) = compute_conflicts_cliques(self.features[:, 6].long())

    def compute_pre_features(self):

        self.total_job_time = np.sum(
            np.where(self.original_durations < 0, 0, self.original_durations), axis=1
        )
        for job_id in range(self.n_jobs):
            if (self.durations[job_id, :, 0] == -1).any():
                self.total_job_time[job_id, 0] = -1
        self.total_job_time = torch.as_tensor(self.total_job_time, dtype=torch.float)

        of = self.features_offset["selectable"]
        self.features[:, of[0] : of[1]] = 0
        for j in range(self.n_jobs):
            if self.affectations[j, 0] != -1:
                self.features[
                    job_and_task_to_node(j, 0, self.max_n_machines), of[0] : of[1]
                ] = 1

        if "total_job_time" in self.features_offset:
            tjtof = self.features_offset["total_job_time"]
            bc = np.broadcast_to(
                self.total_job_time[:, None, :], (self.n_jobs, self.max_n_machines, 4)
            ).reshape((self.n_jobs * self.max_n_machines, 4))
            self.features[:, tjtof[0] : tjtof[1]] = torch.as_tensor(bc)

        self.total_machine_time = torch.zeros((self.n_machines, 4))
        for job_id in range(self.n_jobs):
            for task_id in range(self.n_machines):
                if self.affectations[job_id, task_id] != -1:
                    if (
                        self.total_machine_time[self.affectations[job_id, task_id]][0]
                        == -1
                        or self.durations[job_id, task_id][0] == -1
                    ):
                        self.total_machine_time[
                            self.affectations[job_id, task_id]
                        ] += self.durations[job_id, task_id]
                        self.total_machine_time[self.affectations[job_id, task_id]][
                            0
                        ] = -1
                    else:
                        self.total_machine_time[
                            self.affectations[job_id, task_id]
                        ] += self.durations[job_id, task_id]

        self.total_machine_time_job_task = torch.zeros(
            (self.n_jobs, self.n_machines, 4)
        )
        for job_id in range(self.n_jobs):
            for task_id in range(self.n_machines):
                self.total_machine_time_job_task[
                    job_id, task_id
                ] = self.total_machine_time[self.affectations[job_id, task_id]]
                if "total_machine_time" in self.features_offset:
                    tmtof = self.features_offset["total_machine_time"]
                    self.features[
                        job_and_task_to_node(job_id, task_id, self.max_n_machines),
                        tmtof[0] : tmtof[1],
                    ] = self.total_machine_time[self.affectations[job_id, task_id]]

        self.job_completion_time = torch.zeros((self.n_jobs, 4))

        if "job_completion_percentage" in self.features_offset:
            tjpof = self.features_offset["job_completion_percentage"]
            jcp = self.job_completion_time / self.total_job_time
            jcp = torch.where(self.total_job_time < 0, torch.Tensor([-1.0]), jcp)
            jcpb = (
                jcp.unsqueeze_(1)
                .expand((self.n_jobs, self.max_n_machines, 4))
                .reshape((self.n_jobs * self.max_n_machines, 4))
            )
            self.features[:, tjpof[0] : tjpof[1]] = jcpb

        self.machine_completion_time = torch.zeros((self.n_machines, 4))

        self.machine_completion_time_job_task = torch.zeros(
            (self.n_jobs, self.n_machines, 4)
        )
        for job_id in range(self.n_jobs):
            for task_id in range(self.n_machines):
                self.machine_completion_time_job_task[
                    job_id, task_id
                ] = self.machine_completion_time[self.affectations[job_id, task_id]]
                if "machine_completion_percentage" in self.features_offset:
                    mcpof = self.features_offset["machine_completion_percentage"]
                    result = (
                        self.machine_completion_time_job_task[job_id, task_id]
                        / self.total_machine_time_job_task[job_id, task_id]
                    )
                    result[result != result] = 0
                    self.features[
                        job_and_task_to_node(job_id, task_id, self.max_n_machines),
                        mcpof[0] : mcpof[1],
                    ] = result
                    if self.total_machine_time_job_task[job_id, task_id][0] < 0:
                        self.features[
                            job_and_task_to_node(job_id, task_id, self.max_n_machines),
                            mcpof[0],
                        ] = -1

        if "mopnr" in self.features_offset:
            mopnr = np.sum(self.affectations != -1, axis=1)
            mopnr = np.broadcast_to(
                mopnr[:, None], (self.n_jobs, self.max_n_machines)
            ).flatten()
            self.features[:, self.features_offset["mopnr"][0]] = torch.as_tensor(mopnr)

        if "mwkr" in self.features_offset:
            mwkr = self.total_job_time - self.job_completion_time
            mwkr = (
                mwkr.unsqueeze_(1)
                .expand((self.n_jobs, self.max_n_machines, 4))
                .reshape((self.n_jobs * self.max_n_machines, 4))
            )
            of = self.features_offset["mwkr"]
            self.features[:, of[0] : of[1]] = mwkr

        if "one_hot_job_id" in self.features_offset:
            ohji = np.zeros((self.max_n_jobs, self.max_n_jobs))  # vector of size max_
            for i in range(self.max_n_jobs):
                ohji[i][i] = 1
            ohji = np.broadcast_to(
                ohji[:, None, :],
                (self.max_n_jobs, self.max_n_machines, self.max_n_jobs),
            )
            ohji = np.reshape(
                ohji, (self.max_n_jobs * self.max_n_machines, self.max_n_jobs)
            )
            of = self.features_offset["one_hot_job_id"]
            self.features[:, of[0] : of[1]] = torch.as_tensor(ohji)

    def done(self):
        return np.all(self.affected[np.where(self.affectations >= 0)] > 0)

    def init_features_offset(self, input_list):
        self.features_offset = {}
        self.features_offset["is_affected"] = [0, 1]
        self.features_offset["tct"] = [1, 5]
        self.features_offset["selectable"] = [5, 6]
        self.features_offset["one_hot_machine_id"] = [6, 6 + self.max_n_machines]
        n = 6 + self.max_n_machines
        for input_name in input_list:
            if input_name in [
                "is_affected",
                "completion_time",
                "one_hot_machine_id",
                "selectable",
            ]:
                continue  # already appended above
            if input_name == "one_hot_job_id":
                self.features_offset[input_name] = [n, n + self.max_n_jobs]
                n += self.max_n_jobs
            elif input_name == "mopnr":
                self.features_offset[input_name] = [n, n + 1]
                n += 1
            else:
                self.features_offset[input_name] = [n, n + 4]
                n += 4

    def normalize_features(self, normalize):
        if not normalize:
            return self.features.clone()
        else:
            features = self.features.clone()
            features[:, 1:5] /= self.max_completion_time
            features[:, 1] = torch.where(
                features[:, 1] < 0, torch.Tensor([-1.0]), features[:, 1]
            )
            try:
                dof = self.features_offset["duration"]
                features[:, dof[0] : dof[1]] /= self.max_duration
                features[:, dof[0]] = torch.where(
                    features[:, dof[0]] < 0, torch.Tensor([-1.0]), features[:, dof[0]]
                )
            except KeyError:
                pass
            # if "total_job_time" in self.features_offset:
            try:
                tjo = self.features_offset["total_job_time"]
                features[:, tjo[0] : tjo[1]] /= self.max_completion_time
                features[:, tjo[0]] = torch.where(
                    features[:, tjo[0]] < 0, torch.Tensor([-1.0]), features[:, tjo[0]]
                )
            except KeyError:
                pass

            # if "total_machine_time" in self.features_offset:
            try:
                tmo = self.features_offset["total_machine_time"]
                features[:, tmo[0] : tmo[1]] /= self.max_completion_time
                features[:, tmo[0]] = torch.where(
                    features[:, tmo[0]] < 0, torch.Tensor([-1.0]), features[:, tmo[0]]
                )
            except KeyError:
                pass

            # if "mopnr" in self.features_offset:
            try:
                features[:, self.features_offset["mopnr"][0]] /= self.n_machines
            except KeyError:
                pass

            # if "mwkr" in self.features_offset:
            try:
                of = self.features_offset["mwkr"]
                features[:, of[0] : of[1]] /= self.max_completion_time
                features[:, of[0] : of[1]] = torch.where(
                    features[:, of[0] : of[1]] < 0,
                    torch.Tensor([-1]),
                    features[:, of[0] : of[1]],
                )
            except KeyError:
                pass

            return features

    def to_features_and_edge_index(self, normalize_input, input_list):
        """
        Returns self.graph under the form of a feature and edge_index tensors.
        Note, input_set can contains the following str: 'one_hot_machine_id','one_hot_job_id',
        'duration', 'total_job_time', 'total_machine_time', 'job_completion_percentage',
        'machine_completion_percentage', 'mopnr', 'mwkr'
        """

        features = self.normalize_features(normalize_input)

        if not self.edges:
            edge_index = np.transpose(self.numpy_edges)
        else:
            edge_index = np.transpose(
                np.concatenate([self.numpy_edges, np.array(self.edges)])
            )

        if self.observe_conflicts_as_cliques:
            return (
                features,
                edge_index,
                self.conflicts_edges,
                self.conflicts_edges_machineid,
            )
        return features, edge_index

    def observe_real_duration(
        self, node_id, do_update=True, update_duration_with_real=True
    ):
        job_id, task_id = node_to_job_and_task(node_id, self.n_machines)
        self.is_observed[job_id, task_id] = 1
        self.durations[job_id, task_id][:] = self.original_durations[job_id, task_id][0]
        if "duration" in self.features_offset:
            dof = self.features_offset["duration"]
            if update_duration_with_real:
                # replace min max mode with real duration
                self.features[node_id, dof[0] : dof[1]] = self.original_durations[
                    job_id, task_id
                ][0]
            else:
                # only put real duration, for computing makespan during propagation
                self.features[node_id, dof[0] : dof[0] + 1] = self.original_durations[
                    job_id, task_id
                ][0]

        if do_update:
            self.update_completion_times(node_id)

    def update_completion_times_from(self, node_id):
        # check if node is already solved
        completion_times = self.get_task_completion_times(node_id)
        if max(completion_times) != -1:
            return completion_times

        # recursively solve the predecessors
        predecessors = list(self.graph.predecessors(node_id))
        task_comp_time_pred = torch.stack(
            [self.update_completion_times_from(p) for p in predecessors]
        )
        # The max completion time of predecessors is given by max for each features (real, min, max, and mode)
        # max_completion_time_predecessors = torch.max(task_comp_time_pred, 0)[0]
        ii = task_comp_time_pred.argmax(0)
        max_completion_time_predecessors = task_comp_time_pred.gather(
            0, ii.unsqueeze(0)
        ).squeeze(0)

        # For the real time, if one of the predecessors has an undefined end time, current node is also undefined
        if -1 in task_comp_time_pred:
            max_completion_time_predecessors[0] = -1

        new_completion_time = max_completion_time_predecessors + self.get_durations(
            node_id
        )
        # If there is any uncertainty, we remove the real duration value
        if (
            max_completion_time_predecessors[0] == -1
            or self.is_observed[node_to_job_and_task(node_id, self.n_machines)] == 0
        ):
            new_completion_time[0] = -1

        # update the node (now solved) and return to successor
        self.set_task_completion_times(node_id, new_completion_time)
        return new_completion_time

    def update_completion_times_from_sinks(self):
        sinks = []
        nodes = list(self.graph.nodes)
        for n in nodes:
            successors = list(self.graph.successors(n))
            if len(successors) == 0:
                sinks.append(n)
            # reset all completion times for memoization
            completion_times = torch.ones(4) * -1
            # set completion times for source nodes
            predecessors = list(self.graph.predecessors(n))
            if len(predecessors) == 0:
                completion_times = self.get_durations(n)
            self.set_task_completion_times(n, completion_times)
        for sink in sinks:
            self.update_completion_times_from(sink)

    def update_completion_times_in_order(self):
        priority_queue = PriorityQueue()
        nodes = list(self.graph.nodes)
        for n in nodes:
            predecessors = list(self.graph.predecessors(n))
            if not predecessors:
                priority_queue.put((0, n))
        while not priority_queue.empty():
            (distance, cur_node_id) = priority_queue.get()
            predecessors = list(self.graph.predecessors(cur_node_id))

            if len(predecessors) == 0:
                max_completion_time_predecessors = torch.zeros(4)
            else:
                task_comp_time_pred = torch.stack(
                    [self.get_task_completion_times(p) for p in predecessors]
                )
                # The max completion time of predecessors is given by max for each features (real, min, max, and mode)
                max_completion_time_predecessors = torch.max(task_comp_time_pred, 0)[0]
                # For the real time, if one of the predecessors has an undefined end time, current node is also undefined
                if -1 in task_comp_time_pred:
                    max_completion_time_predecessors[0] = -1

            new_completion_time = max_completion_time_predecessors + self.get_durations(
                cur_node_id
            )
            # If there is any uncertainty, we remove the real duration value
            if (
                max_completion_time_predecessors[0] == -1
                or self.is_observed[node_to_job_and_task(cur_node_id, self.n_machines)]
                == 0
            ):
                new_completion_time[0] = -1

            self.set_task_completion_times(cur_node_id, new_completion_time)
            # Force update all nodes for external use
            for successor in self.graph.successors(cur_node_id):
                priority_queue.put((distance + 1, successor))

    def update_completion_times(self, node_id, rec=True):
        """
        This function is supposed to update the starting time of the selected node
        and all of its succesors. To do so, it travels through the whole graph of
        successors, ordered by their distance to the original node, choosing each time
        the max completion time of predecessors as starting time
        """
        priority_queue = PriorityQueue()
        priority_queue.put((0, node_id))

        while not priority_queue.empty():
            (distance, cur_node_id) = priority_queue.get()
            predecessors = list(self.graph.predecessors(cur_node_id))

            if len(predecessors) == 0:
                max_completion_time_predecessors = torch.zeros(4)
            else:
                task_comp_time_pred = torch.stack(
                    [self.get_task_completion_times(p) for p in predecessors]
                )
                # The max completion time of predecessors is given by max for each features (real, min, max, and mode)
                # max_completion_time_predecessors = torch.max(task_comp_time_pred, 0)[0]
                ii = task_comp_time_pred.argmax(0)
                max_completion_time_predecessors = task_comp_time_pred.gather(
                    0, ii.unsqueeze(0)
                ).squeeze(0)
                # For the real time, if one of the predecessors has an undefined end time, current node is also undefined
                if -1 in task_comp_time_pred:
                    max_completion_time_predecessors[0] = -1

            new_completion_time = max_completion_time_predecessors + self.get_durations(
                cur_node_id
            )
            # If there is any uncertainty, we remove the real duration value
            if (
                max_completion_time_predecessors[0] == -1
                or self.is_observed[node_to_job_and_task(cur_node_id, self.n_machines)]
                == 0
            ):
                new_completion_time[0] = -1

            if rec:
                old_completion_time = self.get_task_completion_times(
                    cur_node_id
                ).clone()
            self.set_task_completion_times(cur_node_id, new_completion_time)

            # Only add the nodes in the queue if update is necessary
            if rec and not torch.equal(old_completion_time, new_completion_time):
                for successor in self.graph.successors(cur_node_id):
                    priority_queue.put((distance + 1, successor))

    def set_precedency(self, first_node_id, second_node_id, do_update=True):
        """
        Check if possible to add an edge between first_node and second_node. Then add it
        and updates all other attributes of the State related to the graph.
        """
        # First check that second_node is not scheduled before first node
        nodes_after_second_node = nx.algorithms.descendants(self.graph, second_node_id)
        if first_node_id in nodes_after_second_node:
            return False
        # Also check that first and second node ids are not the same
        if first_node_id == second_node_id:
            return False
        # Then add the node into the graph
        self.graph.add_edge(first_node_id, second_node_id)
        self.edges.append((first_node_id, second_node_id))

        # Finally update the task starting times
        if do_update:
            self.update_completion_times(second_node_id)
        return True

    def remove_precedency(self, first_node_id, second_node_id):
        self.graph.remove_edge(first_node_id, second_node_id)
        self.edges.remove((first_node_id, second_node_id))
        return True

    def node_same_job(self, jid):
        if jid in self.same_job:
            return self.same_job[jid]

        sj = []
        for t in range(self.max_n_machines):
            if self.affectations[jid, t] != -1:
                sj.append(job_and_task_to_node(jid, t, self.max_n_machines))
            self.same_job[jid] = sj
        return sj

    def on_machine(self, machine_id):
        coord = np.asarray(self.affectations == machine_id).nonzero()
        return [
            job_and_task_to_node(j[0], j[1], self.max_n_machines)
            for j in zip(coord[0], coord[1])
        ]

    def affect_node(self, node_id):
        """
        Sets the self.is_affected to 1 for the current node_id.
        Note : The consistency of this operation is key for the get_machine_availability
        function to work well. This consitency is left to the user of the State class,
        for the moment. Later on, it is important to check this consistency in the
        affect_node function
        """

        job_id, task_id = node_to_job_and_task(node_id, self.n_machines)
        machine_id = self.affectations[job_id, task_id]

        if machine_id != -1:

            if (
                "job_completion_percentage" in self.features_offset
                or "total_job_time" in self.features_offset
                or "mopnr" in self.features_offset
                or "mwkr" in self.features_offset
            ):
                node_same_job = self.node_same_job(job_id)

            if (
                "total_machine_time" in self.features_offset
                or "machine_completion_percentage" in self.features_offset
            ):
                on_machine = self.on_machine(machine_id)

            self.features[node_id, self.features_offset["is_affected"][0]] = 1
            self.affect(node_id)

            if (
                "mwkr" in self.features_offset
                or "job_completion_percentage" in self.features_offset
            ):
                if self.job_completion_time[job_id][0] < 0:
                    self.job_completion_time[job_id][0] = self.get_durations(node_id)[0]
                else:
                    self.job_completion_time[job_id][0] += self.get_durations(node_id)[
                        0
                    ]

            if (
                "mwkr" in self.features_offset
                or "total_job_time" in self.features_offset
                or "job_completion_percentage" in self.features_offset
            ):
                if self.total_job_time[job_id][0] < 0:
                    self.total_job_time[job_id][0] = self.get_durations(node_id)[0]
                else:
                    self.total_job_time[job_id][0] += self.get_durations(node_id)[0]

            if "job_completion_percentage" in self.features_offset:
                tjpof = self.features_offset["job_completion_percentage"]
                for nid in node_same_job:
                    self.features[
                        nid, self.features_offset["job_completion_percentage"][0]
                    ] = (
                        self.job_completion_time[job_id][0]
                        / self.total_job_time[job_id][0]
                    )

            if "total_job_time" in self.features_offset:
                for nid in node_same_job:
                    self.features[
                        nid, self.features_offset["total_job_time"][0]
                    ] = self.total_job_time[job_id][0]

            if (
                "total_machine_time" in self.features_offset
                or "machine_completion_percentage" in self.features_offset
            ):
                for nid in on_machine:
                    jid, tid = node_to_job_and_task(nid, self.max_n_machines)
                    if self.total_machine_time_job_task[jid, tid][0] < 0:
                        self.total_machine_time_job_task[jid, tid][
                            0
                        ] = self.get_durations(node_id)[0]
                    else:
                        self.total_machine_time_job_task[jid, tid][
                            0
                        ] += self.get_durations(node_id)[0]

                if self.total_machine_time[machine_id][0] < 0:
                    self.total_machine_time[machine_id][0] = self.get_durations(
                        node_id
                    )[0]
                else:
                    self.total_machine_time[machine_id][0] += self.get_durations(
                        node_id
                    )[0]

            if "total_machine_time" in self.features_offset:
                for nid in on_machine:
                    self.features[
                        nid, self.features_offset["total_machine_time"][0]
                    ] = self.total_machine_time[machine_id][0]

            if "machine_completion_percentage" in self.features_offset:
                self.machine_completion_time[machine_id] += self.get_durations(node_id)
                same_machine = np.asarray(self.affectations == machine_id).nonzero()
                for coord in zip(same_machine[0], same_machine[1]):
                    self.machine_completion_time_job_task[
                        coord[0], coord[1]
                    ] += self.get_durations(node_id)
                if self.get_durations(node_id)[0] == -1:
                    self.job_completion_time[job_id][0] = -1
                    self.machine_completion_time[machine_id][0] = -1
                    self.machine_completion_time_job_task[job_id, task_id, 0] = -1
                for nid in on_machine:
                    mcpo = self.features_offset["machine_completion_percentage"]
                    self.features[nid, mcpo[0] : mcpo[1]] = (
                        self.machine_completion_time[machine_id]
                        / self.total_machine_time[machine_id]
                    )

            if "mopnr" in self.features_offset:
                for nid in node_same_job:
                    self.features[nid, self.features_offset["mopnr"][0]] -= 1

            if "mwkr" in self.features_offset:
                of = self.features_offset["mwkr"]
                for nid in node_same_job:
                    self.features[nid, of[0] : of[1]] = (
                        self.total_job_time[job_id] - self.job_completion_time[job_id]
                    )

            self.features[node_id, self.features_offset["selectable"][0]] = 0
            for successor in self.graph.successors(node_id):
                sjid, stid = node_to_job_and_task(successor, self.n_machines)
                if self.affectations[sjid, stid] != -1:
                    parents = list(self.graph.predecessors(successor))
                    parents_affected = self.features[
                        parents, self.features_offset["is_affected"][0]
                    ]
                    if torch.all(parents_affected.flatten() == 1):
                        self.features[
                            successor, self.features_offset["selectable"][0]
                        ] = 1

    def get_selectable(self):
        return self.features[:, self.features_offset["selectable"][0]]

    def get_last_task_on_machine(self, machine_id, metric):
        """
        Returns a list of occupancy period on the wanted machine, under the form
        (occupancy_start_time, occupancy_duration, node_id)
        """

        if machine_id in self.last_task_on_machine:
            return self.last_task_on_machine[machine_id]
        return None

        if metric == "realistic":
            index = 0
        elif metric == "optimistic":
            index = 1
        elif metric == "pessimistic":
            index = 2
        elif metric == "averagistic":
            index = 3
        else:
            raise Exception("Metric for machine_occupancy not recognized")
        node_ids = self.on_machine(machine_id)
        last = None
        last_start_time = None

        for node_id in node_ids:
            job_id, task_id = node_to_job_and_task(node_id, self.n_machines)
            is_affected = self.is_affected_by_node(node_id)
            if is_affected == 1:
                duration = self.get_durations(node_id)
                tct = self.get_task_completion_times(node_id)[index].item()
                if tct == -1 or duration[index].item() == -1:
                    start_time = -1
                    raise Exception(
                        "get_machine_occupancy not supported for not observed metric. Please use averagistic"
                    )
                else:
                    start_time = tct - duration[index].item()
                if last is None or (start_time > last_start_time):
                    last = node_id
                    last_start_time = start_time
        return last

    def get_machine_occupancy(self, machine_id, metric):
        """
        Returns a list of occupancy period on the wanted machine, under the form
        (occupancy_start_time, occupancy_duration, node_id)
        """
        if metric == "realistic":
            index = 0
        elif metric == "optimistic":
            index = 1
        elif metric == "pessimistic":
            index = 2
        elif metric == "averagistic":
            index = 3
        else:
            raise Exception("Metric for machine_occupancy not recognized")
        node_ids = self.on_machine(machine_id)
        occupancy = []
        for node_id in node_ids:
            job_id, task_id = node_to_job_and_task(node_id, self.n_machines)
            is_affected = self.is_affected(job_id, task_id)
            if is_affected == 1:
                duration = self.get_durations(node_id)
                tct = self.get_task_completion_times(node_id)[index].item()
                if tct == -1 or duration[index].item() == -1:
                    start_time = -1
                    raise Exception(
                        "get_machine_occupancy not supported for not observed metric. Please use averagistic"
                    )
                else:
                    start_time = tct - duration[index].item()
                occupancy.append((start_time, duration[index].item(), node_id))
        occupancy.sort()
        return occupancy

    def get_solution(self):
        if not self.done():
            return False
        tct = (
            self.features[:, 1]
            .reshape((self.n_jobs, self.max_n_machines, 1))
            .squeeze_(2)
            .numpy()
        )
        schedule = tct - self.original_durations[:, :, 0]
        # we give schedule for real observed durations
        return Solution(
            schedule=schedule, real_durations=self.original_durations[:, :, 0]
        )

    def get_first_unaffected_task(self, job_id):
        """
        Returns the id of the first task that wasn't affected. If all tasks are
        affected, returns -1
        """
        # unaff = np.where(
        #     np.logical_and(self.affected[job_id] == 0, self.affectations[job_id] != -1)
        # )[0]
        unaff = np.where(self.affected[job_id] == 0)[0]

        if unaff.size == 0:
            return None
        return unaff[0]

    def get_job_availability(self, job_id, task_id, metric):
        if task_id == 0:
            return 0
        if metric == "realistic":
            index = 0
        elif metric == "optimistic":
            index = 1
        elif metric == "pessimistic":
            index = 2
        elif metric == "averagistic":
            index = 3
        else:
            raise Exception("Metric for job_availability not recognized")
        tct = self.features[
            job_and_task_to_node(job_id, task_id - 1, self.max_n_machines)
        ]
        return tct[index].item()

    def generate_colors(self):
        n = self.n_machines
        p = 0
        while p * p * p < n:
            p += 1
        scale = [(i / p) + 1 / (2 * p) for i in range(p)]
        colors = [(si, sj, sk) for si in scale for sj in scale for sk in scale]
        return tuple([color for color in colors])

    def render_solution(self, schedule, scaling=1.0):
        df = []
        all_finish = schedule * scaling + self.original_durations[:, :, 0]
        for job in range(self.n_jobs):
            i = 0
            while i < self.n_machines:
                if self.affectations[job][i] == -1:
                    i += 1
                    continue
                dict_op = dict()
                dict_op["Task"] = "Job {}".format(job)
                start_sec = schedule[job][i] * scaling
                finish_sec = all_finish[job][i]
                start_sec += 315532800
                finish_sec += 315532800
                dict_op["Start"] = datetime.datetime.fromtimestamp(start_sec)
                dict_op["Finish"] = datetime.datetime.fromtimestamp(finish_sec)
                dict_op["Resource"] = "Machine {}".format(self.affectations[job][i])
                df.append(dict_op)
                i += 1
        fig = None
        if len(df) > 0:
            df = pd.DataFrame(df)
            fig = ff.create_gantt(
                df,
                index_col="Resource",
                colors=self.colors,
                show_colorbar=True,
                group_tasks=True,
            )
            if fig is not None:
                fig.update_yaxes(
                    autorange="reversed"
                )  # otherwise tasks are listed from the bottom
                figimg = fig.to_image(format="png")
                npimg = np.fromstring(figimg, dtype="uint8")
                cvimg = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
                npimg = np.transpose(cvimg, (2, 0, 1))
                torchimg = torch.from_numpy(npimg)
                return torchimg
            else:
                return None
        else:
            return None

    def display(self, fname="state.png"):
        print("affectation\n", self.affectations)
        # plt.clf()
        # pos = {}
        # for j in range(0, self.n_jobs):
        #     for m in range(0, self.n_machines):
        #         pos[j * self.n_machines + m] = (m, -j)

        print("task_completion_times\n", self.features[:, 1:5])

        # print("machine id", self.features[:, 6 : 6 + self.max_n_machines])

        if "duration" in self.features_offset:
            dof = self.features_offset["duration"]
            print("durations", self.features[:, dof[0] : dof[1]])

        # if "monpr" in self.features_offset:
        #     of = self.features_offset["mopnr"]
        #     print("mopnr", self.features[:, of[0] : of[1]])

        # if "mwkr" in self.features_offset:
        #     of = self.features_offset["mwkr"]
        #     print("mwkr", self.features[:, of[0] : of[1]])

        # print("isaffected\n", self.affected)
        # print("is observed\n", self.is_observed)
        # print("drawing graph")
        # nx.draw_networkx(self.graph, pos, with_labels=True)
        # # nx.draw_planar(self.graph, with_labels=True)
        # plt.savefig(fname)
