from copy import deepcopy
from queue import PriorityQueue

import networkx as nx
import numpy as np
import torch
import torch_geometric

import datetime
import random
import pandas as pd
import plotly.figure_factory as ff
import cv2

from problem.solution import Solution
from utils.utils import node_to_job_and_task, job_and_task_to_node


class State:
    def __init__(self, affectations, durations, max_n_jobs, max_n_machines, deterministic=True, node_encoding="L2D"):
        self.affectations = affectations
        self.original_durations = durations
        self.n_jobs = self.affectations.shape[0]
        self.n_machines = self.affectations.shape[1]
        self.n_nodes = self.n_jobs * self.n_machines

        self.max_n_jobs = max_n_jobs
        self.max_n_machines = max_n_machines
        self.node_encoding = node_encoding
        assert self.node_encoding in ["L2D", "DenseL2D"]
        self.deterministic = deterministic

        self.colors = self.generate_colors()

        if self.node_encoding == "DenseL2D":
            self.return_graph = None
        self.graph = None

        self.task_completion_times = None
        self.is_affected = None
        self.is_observed = None
        self.durations = None
        self.n_jobs_per_machine = None
        self.n_machines_per_job = None

        # Used to compute the features
        self.max_duration = None
        self.max_completion_time = None
        self.total_job_time = None
        self.total_machine_time = None
        self.job_completion_time = None
        self.machine_completion_time = None
        self.number_operations_scheduled = None

        self.reset()

    def reset(self):
        self.graph = nx.DiGraph(
            [
                (
                    job_index * self.n_machines + i,
                    job_index * self.n_machines + i + 1,
                )
                for i in range(self.n_machines - 1)
                for job_index in range(self.n_jobs)
            ]
        )
        if self.node_encoding == "DenseL2D":
            self.return_graph = deepcopy(self.graph)
            for machine_id in range(self.n_machines):
                node_ids = self._get_machine_node_ids(machine_id)
                for first_node_id in node_ids:
                    for second_node_id in node_ids:
                        if second_node_id != first_node_id:
                            self.return_graph.add_edge(first_node_id, second_node_id)

        # Instantiate features and pre features
        self.durations = self.original_durations.copy()
        self.task_completion_times = np.cumsum(self.durations, axis=1)
        if not self.deterministic:
            self.durations[:, :, 0] = -1
            self.task_completion_times[:, :, 0] = -1
        self.is_affected = np.zeros_like(self.affectations)
        self.is_observed = np.ones_like(self.affectations) if self.deterministic else np.zeros_like(self.affectations)
        self.n_jobs_per_machine = np.array([(self.affectations == m).sum() for m in range(self.n_machines)])
        self.n_machines_per_job = np.array(
            [self.n_machines - (self.affectations[j] == -1).sum() for j in range(self.n_jobs)]
        )
        self.number_operations_scheduled = np.zeros(self.n_jobs)
        self.max_duration = np.max(self.durations.flatten())
        self.max_completion_time = np.max(self.task_completion_times.flatten())

        self.compute_pre_features()

    def compute_pre_features(self):
        self.task_completion_times = np.cumsum(self.durations, axis=1)
        for job_id in range(self.n_jobs):
            for task_id in range(self.n_machines):
                if self.durations[job_id, task_id, 0] == -1:
                    self.task_completion_times[job_id, task_id:, 0] = -1

        self.total_job_time = np.sum(self.durations, axis=1)
        for job_id in range(self.n_jobs):
            if (self.durations[job_id, :, 0] == -1).any():
                self.total_job_time[job_id, 0] = -1

        self.total_machine_time = np.zeros((self.n_machines, 4))
        for job_id in range(self.n_jobs):
            for task_id in range(self.n_machines):
                if self.affectations[job_id, task_id] != -1:
                    if (
                        self.total_machine_time[self.affectations[job_id, task_id]][0] == -1
                        or self.durations[job_id, task_id][0] == -1
                    ):
                        self.total_machine_time[self.affectations[job_id, task_id]] += self.durations[job_id, task_id]
                        self.total_machine_time[self.affectations[job_id, task_id]][0] = -1
                    else:
                        self.total_machine_time[self.affectations[job_id, task_id]] += self.durations[job_id, task_id]

        self.job_completion_time = np.zeros((self.n_jobs, 4))
        for job_id in range(self.n_jobs):
            for task_id in range(self.n_machines):
                if self.is_affected[job_id, task_id] == 1 and self.affectations[job_id, task_id] != -1:
                    if self.job_completion_time[job_id, 0] == -1 or self.durations[job_id, task_id, 0] == -1:
                        self.job_completion_time[job_id] += self.durations[job_id, task_id]
                        self.job_completion_time[job_id, 0] = -1
                    else:
                        self.job_completion_time[job_id] += self.durations[job_id, task_id]

        self.machine_completion_time = np.zeros((self.n_machines, 4))
        for job_id in range(self.n_jobs):
            for task_id in range(self.n_machines):
                if self.is_affected[job_id, task_id] == 1 and self.affectations[job_id, task_id] != -1:
                    if (
                        self.machine_completion_time[self.affectations[job_id, task_id], 0] == -1
                        or self.durations[job_id, task_id, 0] == -1
                    ):
                        self.machine_completion_time[self.affectations[job_id, task_id]] += self.durations[job_id, task_id]
                        self.machine_completion_time[self.affectations[job_id, task_id], 0] = -1
                    else:
                        self.machine_completion_time[self.affectations[job_id, task_id]] += self.durations[job_id, task_id]

    def done(self):
        """
        The problem is solved when each machine is completely ordered, meaning that we
        know for each machine exactly which job is first, which is second, etc...
        In order to check this, we check that the longest path in every machine subgraph
        contains exactly n-1 edges where n is the number of jobs
        """
        for machine_id in range(self.n_machines):
            machine_sub_graph = self.graph.subgraph(self._get_machine_node_ids(machine_id))
            if nx.algorithms.dag.dag_longest_path_length(machine_sub_graph) != self.n_jobs_per_machine[machine_id] - 1:
                return False
        return True

    def _get_machine_node_ids(self, machine_id):
        node_ids = []
        if machine_id == -1:
            return node_ids
        for job_id in range(self.n_jobs):
            for task_id in range(self.n_machines):
                if self.affectations[job_id, task_id] == machine_id:
                    node_id = job_and_task_to_node(job_id, task_id, self.n_machines)
                    node_ids.append(node_id)
        return node_ids

    def to_features_and_edge_index(self, normalize_input, input_list):
        """
        Returns self.graph under the form of a torch_geometric.data.Data object.
        The node_encoding arguments specifies what are the features (i.e. the x
        parameter of the Data object) that should be added to the graph.
        Note, input_set can contains the following str: 'one_hot_machine_id','one_hot_job_id',
        'duration', 'total_job_time', 'total_machine_time', 'job_completion_percentage',
        'machine_completion_percentage', 'mopnr', 'mwkr'
        """
        for job_id in range(self.n_jobs):
            for task_id in range(self.n_machines):
                machine_id = self.affectations[job_id, task_id]
                node_id = job_and_task_to_node(job_id, task_id, self.n_machines)

                # Compute features
                features = self.get_features(job_id, task_id, machine_id, normalize_input)

                node_vector = (
                    [node_id]
                    + features["is_affected"].tolist()
                    + features["completion_time"].tolist()
                    + features["one_hot_machine_id"].tolist()
                )

                for input_name in input_list:
                    if input_name in ["is_affected", "completion_time", "one_hot_machine_id"]:
                        continue  # already appended above
                    node_vector = node_vector + features[input_name].tolist()

                if self.node_encoding == "L2D":
                    self.graph.nodes[node_id]["x"] = node_vector
                elif self.node_encoding == "DenseL2D":
                    self.return_graph.nodes[node_id]["x"] = node_vector

        nx_graph = self.graph if self.node_encoding == "L2D" else self.return_graph
        graph = torch_geometric.utils.from_networkx(nx_graph)

        # We have to reorder features, since the networx -> torch_geometric
        # shuffles the nodes
        node_ids = graph.x[:, 0].long()
        features = torch.zeros((self.n_nodes, graph.x[:, 1:].shape[1]))
        features[node_ids] = graph.x[:, 1:].float()
        edge_index = node_ids[graph.edge_index]

        return features, edge_index

    def get_features(self, job_id, task_id, machine_id, normalize_input):
        """
        Returns the specified features in a dict.
        The inputs are normalized if normalize_input is set to True
        """
        # Mandatory features
        features = {}
        features["is_affected"] = np.repeat(self.is_affected[job_id, task_id], 4)  # vector of size 4
        features["completion_time"] = self.task_completion_times[job_id, task_id]  # vector of size 4

        # Other features
        features["one_hot_machine_id"] = self.to_one_hot(machine_id, self.max_n_machines)  # vector of size max_n_jobs
        features["one_hot_job_id"] = self.to_one_hot(job_id, self.max_n_jobs)  # vector of size max_n_machines
        features["duration"] = self.durations[job_id, task_id]  # vector of size 4
        features["total_job_time"] = self.total_job_time[job_id]  # vector of size 4
        features["total_machine_time"] = self.total_machine_time[machine_id]  # vector of size 4
        features["job_completion_percentage"] = self.job_completion_time[job_id] / features["total_job_time"]  # size 4
        features["machine_completion_percentage"] = self.machine_completion_time[machine_id] / features["total_machine_time"]
        # Checking consistency with knwoledge
        for i in range(4):
            if self.job_completion_time[job_id][i] == -1 or features["total_job_time"][i] == -1:
                features["job_completion_percentage"][i] = -1
            if self.machine_completion_time[machine_id][i] == -1 or features["total_machine_time"][i] == -1:
                features["machine_completion_percentage"][i] = -1

        # See https://hal.archives-ouvertes.fr/hal-00728900/document for the definition of these metrics
        features["mopnr"] = np.repeat(self.n_machines - self.number_operations_scheduled[job_id], 4)  # vector of size 4
        features["mwkr"] = features["total_job_time"] - self.job_completion_time[job_id]  # vector of size 4
        # Checking consistency with knowledge
        for i in range(4):
            if features["total_job_time"][i] == -1 or self.job_completion_time[job_id][i] == -1:
                features["mwkr"][i] = -1

        if normalize_input:
            features["completion_time"] = features["completion_time"] / self.max_completion_time
            features["duration"] = features["duration"] / self.max_duration
            features["total_job_time"] = features["total_job_time"] / self.max_completion_time
            features["total_machine_time"] = features["total_machine_time"] / self.max_completion_time
            features["mopnr"] = features["mopnr"] / self.n_machines
            features["mwkr"] = features["mwkr"] / self.max_completion_time
            # Checking consistency with knowledge
            features["completion_time"][features["completion_time"] < 0] = -1
            features["duration"][features["duration"] < 0] = -1
            features["total_job_time"][features["total_job_time"] < 0] = -1
            features["total_machine_time"][features["total_machine_time"] < 0] = -1
            features["mwkr"][features["mwkr"] < 0] = -1

        return features

    def to_one_hot(self, index, max_index):
        rep = np.zeros(max_index)
        rep[index] = 1
        return rep

    def observe_real_duration(self, node_id, do_update=True):
        if self.deterministic:
            return
        job_id, task_id = node_to_job_and_task(node_id, self.n_machines)
        self.is_observed[job_id, task_id] = 1
        self.durations[job_id, task_id][0] = self.original_durations[job_id, task_id][0]

        # Re compute pre features using knowledge on durations
        self.compute_pre_features()

        if do_update:
            self._update_completion_times(node_id)

    def _update_completion_times(self, node_id):
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
                max_completion_time_predecessors = np.zeros(4)
            else:
                task_comp_time_pred = np.stack(
                    [self.task_completion_times[node_to_job_and_task(p, self.n_machines)] for p in predecessors]
                )
                # The max completion time of predecessors is given by max for each features (real, min, max, and mode)
                max_completion_time_predecessors = task_comp_time_pred.max(axis=0)
                # For the real time, if one of the predecessors has an undefined end time, current node is also undefined
                if -1 in task_comp_time_pred:
                    max_completion_time_predecessors[0] = -1

            new_completion_time = (
                max_completion_time_predecessors + self.durations[node_to_job_and_task(cur_node_id, self.n_machines)]
            )
            # If there is any uncertainty, we remove the real duration value
            if (
                max_completion_time_predecessors[0] == -1
                or self.is_observed[node_to_job_and_task(cur_node_id, self.n_machines)] == 0
            ):
                new_completion_time[0] = -1
            old_completion_time = self.task_completion_times[node_to_job_and_task(cur_node_id, self.n_machines)].copy()
            self.task_completion_times[node_to_job_and_task(cur_node_id, self.n_machines)] = new_completion_time

            # Only add the nodes in the queue if update is necessary
            if not np.array_equal(old_completion_time, new_completion_time):
                for successor in self.graph.successors(cur_node_id):
                    priority_queue.put((distance + 1, successor))

    def set_precedency(self, first_node_id, second_node_id):
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
        if self.node_encoding == "DenseL2D":
            self.update_return_graph("add_precedency", first_node_id, second_node_id)

        # Finally update the task starting times
        self._update_completion_times(second_node_id)
        return True

    def remove_precedency(self, first_node_id, second_node_id):
        self.graph.remove_edge(first_node_id, second_node_id)
        if self.node_encoding == "DenseL2D":
            self.update_return_graph("remove_precedency", first_node_id, second_node_id)
        return True

    def update_return_graph(self, operation, first_node_id, second_node_id):
        """
        The return graph is updated when the graph is. We update for adding edges, but removing edges do nothing to the
        return graph.
        """
        if self.node_encoding != "DenseL2D":
            return
        if operation == "add_precedency":
            for p in self.graph.predecessors(first_node_id):
                if self.return_graph.has_edge(second_node_id, p):
                    self.return_graph.remove_edge(second_node_id, p)
            for s in self.graph.successors(second_node_id):
                if self.return_graph.has_edge(s, first_node_id):
                    self.return_graph.remove_edge(s, first_node_id)
            if self.return_graph.has_edge(second_node_id, first_node_id):
                self.return_graph.remove_edge(second_node_id, first_node_id)
        elif operation == "remove_precedency":
            pass
        else:
            raise Exception("Operation not recognized")

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
            self.is_affected[job_id, task_id] = 1
            self.job_completion_time[job_id] += self.durations[job_id, task_id]
            self.machine_completion_time[machine_id] += self.durations[job_id, task_id]
            if self.durations[job_id, task_id][0] == -1:
                self.job_completion_time[job_id][0] = -1
                self.machine_completion_time[machine_id][0] = -1
            self.number_operations_scheduled[job_id] += 1

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
        node_ids = self._get_machine_node_ids(machine_id)
        occupancy = []
        for node_id in node_ids:
            job_id, task_id = node_to_job_and_task(node_id, self.n_machines)
            is_affected = self.is_affected[job_id, task_id]
            duration = self.durations[job_id, task_id]
            if is_affected == 1:
                start_time = self.task_completion_times[job_id, task_id][index] - duration[index]
                if self.task_completion_times[job_id, task_id][index] == -1 or duration[index] == -1:
                    start_time = -1
                    raise Exception("get_machine_occupancy not supported for not observed metric. Please use averagistic")
                occupancy.append((start_time, duration[index], node_id))
        occupancy.sort()
        return occupancy

    def get_solution(self):
        if not self.done():
            return False
        schedule = self.task_completion_times[:, :, 0] - self.original_durations[:, :, 0]
        # we give schedule for real observed durations
        return Solution(schedule=schedule, real_durations=self.original_durations[:, :, 0])

    def get_first_unaffected_task(self, job_id):
        """
        Returns the id of the first task that wasn't affected. If all tasks are
        affected, returns -1
        """
        if np.sum(self.is_affected[job_id]) == self.n_machines_per_job[job_id]:
            return -1
        return list(self.is_affected[job_id]).index(0)

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
        return self.task_completion_times[job_id, task_id - 1][index]

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
        all_finish = schedule * scaling + self.durations[:, :, 0]
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
                dict_op["Start"] = datetime.datetime.fromtimestamp(start_sec)
                dict_op["Finish"] = datetime.datetime.fromtimestamp(finish_sec)
                dict_op["Resource"] = "Machine {}".format(self.affectations[job][i])
                df.append(dict_op)
                i += 1
        fig = None
        if len(df) > 0:
            df = pd.DataFrame(df)
            fig = ff.create_gantt(df, index_col="Resource", colors=self.colors, show_colorbar=True, group_tasks=True)
            if fig is not None:
                fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom
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
