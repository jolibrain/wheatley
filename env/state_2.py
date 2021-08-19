import networkx as nx
import numpy as np
import torch_geometric

from utils.utils import node_to_job_and_task


class State:
    def __init__(self, affectations, durations):
        self.affectations = affectations
        self.durations = durations
        self.n_jobs = self.affectations.shape[0]
        self.n_machines = self.affectations.shape[1]

        self.graph = None
        self.task_starting_times = None
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

        self.task_starting_times = -np.ones_like(self.affectations)
        self.task_starting_times[0, :] = 0

    def done(self):
        for machine_id in range(self.n_machines):
            machine_sub_graph = self.graph.subgraph(
                [self._get_machine_node_ids(machine_id)]
            )
            if len(machine_sub_graph.edges) < self.n_jobs:
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
        if node_encoding == "L2D":
            x = np.zeros(self.n_jobs, self.n_machines, 2)
            for i in range(self.n_jobs):
                for j in range(self.n_machines):
                    if self.task_starting_times[i, j] != -1:
                        x[i, j] = [1, self.task_starting_times[i, j]]
                    else:
                        x[i, j] = [
                            0,
                            self.task_starting_times[i, j - 1]
                            + self.durations[i, j],
                        ]

            for node_id in range(self.n_machines * self.n_jobs):
                job_id, task_id = node_to_job_and_task(
                    node_id, self.n_machines
                )
                self.graph[node_id]["x"] = x[job_id, task_id]
            return torch_geometric.utils.from_networkx(self.graph)

        else:
            raise Exception("Encoding not recognized")

    def get_machine_availability(self, machine_id):
        # nodes_id = self._get_machine_nodes_ids(machine_id)
        pass
