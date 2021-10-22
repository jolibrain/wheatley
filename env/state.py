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

from config import MAX_N_MACHINES, MAX_N_JOBS

COLORS = [tuple([random.random() for _ in range(3)]) for _ in range(MAX_N_MACHINES)]


class State:
    def __init__(self, affectations, durations, node_encoding="L2D"):
        self.affectations = affectations
        self.original_durations = durations
        self.n_jobs = self.affectations.shape[0]
        self.n_machines = self.affectations.shape[1]
        self.n_nodes = self.n_jobs * self.n_machines

        if len(COLORS) > self.n_machines:
            self.colors = COLORS[: self.n_machines]
        else:
            self.colors = COLORS

        self.node_encoding = node_encoding
        if self.node_encoding == "DenseL2D":
            self.return_graph = None

        self.graph = None

        self.task_completion_times = None
        self.is_affected = None
        self.is_observed = None
        self.durations = self.original_durations.copy()

        # Used to compute the features
        self.max_duration = None
        self.max_completion_time = None
        self.total_job_time = None
        self.total_machine_time = None
        self.job_completion_time = None
        self.machine_completion_time = None
        self.number_operations_scheduled = None

        self.n_jobs_per_machine = np.zeros(self.n_machines)
        for m in range(0, self.n_machines):
            self.n_jobs_per_machine[m] = (self.affectations == m).sum()
        #print('n_jobs_per_machine=',self.n_jobs_per_machine)
        self.n_machines_per_job = np.zeros(self.n_jobs)
        for j in range(0, self.n_jobs):
            self.n_machines_per_job[j] = self.n_machines - (self.affectations[j] == -1).sum()
        #print('n_machines_per_job=',self.n_machines_per_job)
        
        self.reset()

    def reset(self):
        # durations bounds will be overwritten during solving w/ uncertainty
        self.durations = self.original_durations.copy()
        if self.durations.shape[2] > 1:
            self.durations[:,:,0] = -1
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

        if self.durations.shape[2] == 1:
            self.task_completion_times = np.cumsum(self.durations, axis=1)
        else: # uncertainty case
            self.task_completion_times = np.empty_like(self.durations)
            for i in range(1,self.durations.shape[2]): # do not use real durations drawn from distrib
                self.task_completion_times[:,:,i] = np.cumsum(self.durations[:,:,i], axis=1)
            # real task completion times are not known atm
            self.task_completion_times[:,:,0] = np.zeros_like(self.durations[:,:,0])-1

        self.is_affected = np.zeros_like(self.affectations)
        self.is_observed = np.zeros_like(self.affectations)

        # Used to compute the features
        self.max_duration = np.max(self.durations.flatten())
        self.max_completion_time = np.max(self.task_completion_times.flatten())
        self.total_job_time = np.sum(self.durations, axis=1)
        self.total_machine_time = np.zeros((self.n_machines,self.durations.shape[2]))
        for job_id in range(self.n_jobs):
            for task_id in range(self.n_machines):
                if self.affectations[job_id, task_id] != -1:
                    self.total_machine_time[self.affectations[job_id, task_id]] += self.durations[job_id, task_id]
        self.job_completion_time = np.zeros((self.n_jobs,self.durations.shape[2]))
        self.machine_completion_time = np.zeros((self.n_machines,self.durations.shape[2]))
        self.number_operations_scheduled = np.zeros(self.n_jobs)

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
                node_id = job_and_task_to_node(job_id, task_id, self.n_machines)
                if self.affectations[job_id, task_id] == machine_id:
                    node_ids.append(node_id)
        return node_ids

    def to_torch_geometric(self, normalize_input, input_list):
        """
        Returns self.graph under the form of a torch_geometric.data.Data object.
        The node_encoding arguments specifies what are the features (i.e. the x
        parameter of the Data object) that should be added to the graph.
        Note, input_set can contains the following str: 'is_affected', 'completion_time', 'one_hot_machine_id',
        'one_hot_job_id', 'duration', 'total_job_time', 'total_machine_time', 'job_completion_percentage',
        'machine_completion_percentage', 'mopnr', 'mwkr', 'cr'
        """
        if self.node_encoding in ["L2D", "DenseL2D"]:
            for job_id in range(self.n_jobs):
                for task_id in range(self.n_machines):
                    machine_id = self.affectations[job_id, task_id]

                    node_id = job_and_task_to_node(job_id, task_id, self.n_machines)

                    # Mandatory eatures
                    is_affected = self.is_affected[job_id, task_id]
                    completion_time = self.task_completion_times[job_id, task_id] / (
                        self.max_completion_time if normalize_input else 1
                    )

                    # Other features
                    one_hot_machine_id = self.to_one_hot(machine_id, MAX_N_MACHINES)
                    one_hot_job_id = self.to_one_hot(job_id, MAX_N_JOBS)
                    duration = self.durations[job_id, task_id]
                    total_job_time = self.total_job_time[job_id]
                    total_machine_time = self.total_machine_time[machine_id]
                    job_completion_percentage = self.job_completion_time[job_id] / total_job_time
                    machine_completion_percentage = self.machine_completion_time[machine_id] / total_machine_time
                    original_duration = self.original_durations[job_id, task_id]

                    # See https://hal.archives-ouvertes.fr/hal-00728900/document for the definition of these metrics
                    mopnr = self.n_machines - self.number_operations_scheduled[job_id]
                    mwkr = total_job_time - self.job_completion_time[job_id]
                    cr = duration / (total_job_time - self.job_completion_time[job_id] + self.max_duration)

                    if normalize_input:
                        duration = duration / self.max_duration
                        total_job_time = total_job_time / self.max_completion_time
                        total_machine_time = total_machine_time / self.max_completion_time
                        mopnr = mopnr / self.n_machines
                        mwkr = mwkr / self.max_completion_time

                    node_vector = [node_id, is_affected] + completion_time.tolist()

                    for input_name in input_list:
                        if input_name in ["duration","total_job_time", "total_machine_time",
                                          "job_completion_percentage","machine_completion_percentage",
                                          "mwkr"]:
                            node_vector += locals()[input_name].tolist()
                        elif input_name[0:7] == "one_hot":
                            node_vector = node_vector + locals()[input_name]
                        elif input_name == "is_affected":
                            pass # already appended above
                        elif input_name == "completion_time":
                            pass # already appended above
                        else:
                            node_vector.append(locals()[input_name])

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

            return torch_geometric.data.Data(x=features, edge_index=edge_index)

        else:
            raise Exception("Encoding not recognized")

    def to_one_hot(self, index, max_index):
        rep = [0] * max_index
        rep[index] = 1
        return rep

    def observe_real_duration(self,node_id, do_update = True):
            job_id,task_id = node_to_job_and_task(node_id, self.n_machines)
            self.is_observed[job_id, task_id] = 1
            self.durations[job_id, task_id][:3] = self.original_durations[job_id, task_id][0]
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
            all_pred_known = True
            if len(predecessors) == 0:
                max_completion_time_predecessors = np.zeros(self.task_completion_times.shape[2])
            else:
                task_comp_time_pred = np.stack([self.task_completion_times[node_to_job_and_task(p, self.n_machines)]
                                       for p in predecessors], axis=-1)
                max_completion_time_predecessors = task_comp_time_pred.max(axis = -1)
                if -1 in task_comp_time_pred[0,:]:
                    all_pred_known  = False

            if self.durations.shape[2] == 1:
                new_completion_time =  max_completion_time_predecessors + \
                                       self.durations[node_to_job_and_task(cur_node_id,
                                                                           self.n_machines)]
            else:
                # min max modes are known
                new_completion_time = max_completion_time_predecessors + \
                                      self.durations[node_to_job_and_task(cur_node_id,
                                                                          self.n_machines)]
                # if some uncertainty remains : remove real value
                if not all_pred_known   or \
                   self.is_observed[node_to_job_and_task(cur_node_id, self.n_machines)] == 0:
                    new_completion_time[0] = -1


            old_completion_time = self.task_completion_times[node_to_job_and_task(cur_node_id, self.n_machines)].copy()
            self.task_completion_times[node_to_job_and_task(cur_node_id, self.n_machines)] = new_completion_time

            # Only add the nodes in the queue if update is necessary
            if not np.array_equal(old_completion_time,new_completion_time):
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
        if machine_id == -1:
            #self.is_affected[job_id, task_id] = 1
            return 
        else:
            self.is_affected[job_id, task_id] = 1
            if self.durations.shape[2] > 1:
                self.job_completion_time[job_id] += self.durations[job_id, task_id][3]
                self.machine_completion_time[machine_id] += self.durations[job_id, task_id][3]
            else:
                self.job_completion_time[job_id] += self.durations[job_id, task_id][0]
                self.machine_completion_time[machine_id] += self.durations[job_id, task_id][0]
            self.number_operations_scheduled[job_id] += 1


    def get_machine_occupancy(self, machine_id):
        """
        Returns a list of occupancy period on the wanted machine, under the form
        (occupancy_start_time, occupancy_duration, node_id)
        """
        node_ids = self._get_machine_node_ids(machine_id)
        occupancy = []
        for node_id in node_ids:
            job_id, task_id = node_to_job_and_task(node_id, self.n_machines)
            is_affected = self.is_affected[job_id, task_id]
            if is_affected == 1:
                duration = self.durations[job_id, task_id]
                start_time = self.task_completion_times[job_id, task_id][0] - duration[0]
                occupancy.append((start_time, duration[0], node_id))
        occupancy.sort()
        return occupancy

    def get_solution(self):
        if not self.done():
            return False
        schedule = self.task_completion_times[:,:,0] - self.durations[:,:,0]
        # we give schedule for real observed durations
        return Solution(schedule=schedule)

    def render_solution(self, schedule, scaling=1.0):
        df = []
        all_finish = schedule*scaling + self.durations[:,:,0]

        for job in range(self.n_jobs):
            i = 0
            while i < self.n_machines:
                if self.affectations[job][i] == -1:
                    i += 1
                    continue
                dict_op = dict()
                dict_op["Task"] = "Job {}".format(job)
                start_sec = schedule[job][i]*scaling
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

    def get_first_unaffected_task(self, job_id):
        """
        Returns the id of the first task that wasn't affected. If all tasks are
        affected, returns -1
        """
        if np.sum(self.is_affected[job_id]) == self.n_machines_per_job[job_id]:
            return -1
        return list(self.is_affected[job_id]).index(0)

    def get_job_availability(self, job_id, task_id):
        if task_id == 0:
            return 0
        return self.task_completion_times[job_id, task_id - 1]
