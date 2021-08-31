from queue import PriorityQueue

import networkx as nx
import numpy as np
import torch_geometric

from problem.solution import Solution
from utils.utils import node_to_job_and_task


class State:
    def __init__(self, affectations, durations):
        self.affectations = affectations
        self.durations = durations
        self.n_jobs = self.affectations.shape[0]
        self.n_machines = self.affectations.shape[1]

        self.graph = None
        self.task_completion_times = None
        self.is_affected = None

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

        self.task_completion_times = np.cumsum(self.durations, axis=1)

        self.is_affected = np.zeros_like(self.affectations)

    def done(self):
        """
        The problem is solved when each machine is completely ordered, meaning that we
        know for each machine exactly which job is first, which is second, etc...
        In order to check this, we check that the longest path in every machine subgraph
        contains exactly n-1 edges where n is the number of jobs
        """
        for machine_id in range(self.n_machines):
            machine_sub_graph = self.graph.subgraph(
                self._get_machine_node_ids(machine_id)
            )
            if (
                nx.algorithms.dag.dag_longest_path_length(machine_sub_graph)
                != self.n_jobs - 1
            ):
                return False
        return True

    def _get_machine_node_ids(self, machine_id):
        node_ids = []
        for node_id in range(self.n_jobs * self.n_machines):
            job_id, task_id = node_to_job_and_task(node_id, self.n_machines)
            if self.affectations[job_id, task_id] == machine_id:
                node_ids.append(node_id)
        return node_ids

    def to_torch_geometric(self, node_encoding="L2D"):
        """
        Returns self.graph under the form of a torch_geometric.data.Data object.
        The node_encoding arguments specifies what are the features (i.e. the x
        parameter of the Data object) that should be added to the graph.
        """
        if node_encoding == "L2D":
            for node_id in range(self.n_machines * self.n_jobs):
                job_id, task_id = node_to_job_and_task(
                    node_id, self.n_machines
                )
                self.graph.nodes[node_id]["x"] = [
                    node_id,
                    self.is_affected[job_id, task_id],
                    self.task_completion_times[job_id, task_id],
                ]
            return torch_geometric.utils.from_networkx(self.graph)

        else:
            raise Exception("Encoding not recognized")

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
            max_completion_time_predecessors = (
                max(
                    [
                        self.task_completion_times[
                            node_to_job_and_task(p, self.n_machines)
                        ]
                        for p in predecessors
                    ]
                )
                if len(predecessors) != 0
                else 0
            )

            new_completion_time = (
                max_completion_time_predecessors
                + self.durations[
                    node_to_job_and_task(cur_node_id, self.n_machines)
                ]
            )

            self.task_completion_times[
                node_to_job_and_task(cur_node_id, self.n_machines)
            ] = new_completion_time

            for successor in self.graph.successors(cur_node_id):
                priority_queue.put((distance + 1, successor))

    def set_precedency(self, first_node_id, second_node_id):
        """
        Check if possible to add an edge between first_node and second_node. Then add it
        and updates all other attributes of the State related to the graph.
        """
        # First check that second_node is not scheduled before first node
        nodes_after_second_node = nx.algorithms.descendants(
            self.graph, second_node_id
        )
        if first_node_id in nodes_after_second_node:
            return False
        # Also check that first and second node ids are not the same
        if first_node_id == second_node_id:
            return False
        # Then add the node into the graph
        self.graph.add_edge(first_node_id, second_node_id)
        # Finally update the task starting times
        self._update_completion_times(second_node_id)
        return True

    def affect_node(self, node_id):
        """
        Sets the self.is_affected to 1 for the current node_id.
        Note : The consistency of this operation is key for the get_machine_availability
        function to work well. This consitency is left to the user of the State class,
        for the moment. Later on, it is important to check this consistency in the
        affect_node function
        """
        self.is_affected[node_to_job_and_task(node_id, self.n_machines)] = 1

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
            duration = self.durations[job_id, task_id]
            start_time = self.task_completion_times[job_id, task_id] - duration
            if is_affected == 1:
                occupancy.append((start_time, duration, node_id))
        occupancy.sort()
        return occupancy

    def get_solution(self):
        if not self.done():
            return False
        schedule = self.task_completion_times - self.durations
        return Solution(schedule=schedule)

    def get_first_unaffected_task(self, job_id):
        """
        Returns the id of the first task that wasn't affected. If all tasks are
        affected, returns -1
        """
        if np.sum(self.is_affected[job_id]) == self.n_machines:
            return -1
        return list(self.is_affected[job_id]).index(0)

    def get_job_availability(self, job_id, task_id):
        if task_id == 0:
            return 0
        return self.task_completion_times[job_id, task_id - 1]