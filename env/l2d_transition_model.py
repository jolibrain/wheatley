import torch

from env.transition_model import TransitionModel
from utils.utils import job_and_task_to_node, node_to_job_and_task

from config import MAX_N_EDGES, MAX_N_NODES


class L2DTransitionModel(TransitionModel):
    def __init__(self, affectations, durations):
        super(L2DTransitionModel, self).__init__(affectations, durations, node_encoding="L2D")
        self.useless_timesteps = 0

    def run(self, first_node_id, second_node_id):
        # Since L2D operates on nodes, and not edges, each action must correspond to
        # an edge with the same node on both sides
        if first_node_id != second_node_id:
            self.useless_timesteps += 1
            return
        node_id = first_node_id

        job_id, task_id = node_to_job_and_task(node_id, self.n_machines)
        # To be a valid transition model, the L2DTransitionModel must accept every
        # possible edge_id. But, for most of them, it doesn't do anything, since they
        # don't correspond to any valid action. The model shouldn't actually propose
        # such actions, since we apply a mask to allow only valid actions to happen.
        # Here, we check that the proposed action is a valid one.

        # If the job_id is bigger that max job_id, we don't operate the action
        if job_id >= self.n_jobs:
            self.useless_timesteps += 1
            return
        # Finally, if the task doesn't correspond to the available tasks, we also skip
        if task_id != self.state.get_first_unaffected_task(job_id):
            self.useless_timesteps += 1
            return

        machine_id = self.affectations[job_id, task_id]
        machine_occupancy = self.state.get_machine_occupancy(machine_id)
        job_availability_time = self.state.get_job_availability(job_id, task_id)

        # If no task is affected on machine, just affect it wherever possible and returns
        if not machine_occupancy:
            self.state.affect_node(node_id)
            return

        # Checks wheter task is inserted at the begining, in between or at the end
        if job_availability_time < machine_occupancy[0][0]:
            # Insert task before all other tasks
            self.state.set_precedency(node_id, machine_occupancy[0][2])

        else:
            # Find where there are free times, and check if we can insert task
            index = -1
            for i in range(len(machine_occupancy) - 1):
                start_time, duration, _ = machine_occupancy[i]
                next_start_time, next_duration, _ = machine_occupancy[i + 1]
                if start_time + duration < next_start_time:
                    if start_time + duration <= job_availability_time and job_availability_time < next_start_time:
                        index = i
            if index == -1:
                # The job can be inserted nowhere, so we add it at the end
                self.state.set_precedency(machine_occupancy[-1][2], node_id)
            else:
                # The job is inserted between task_index and task_index+1
                self.state.set_precedency(machine_occupancy[index][2], node_id)
                self.state.set_precedency(node_id, machine_occupancy[index + 1][2])
        self.state.affect_node(node_id)

    def get_mask(self):
        available_node_ids = []
        for job_id in range(self.n_jobs):
            task_id = self.state.get_first_unaffected_task(job_id)
            if task_id != -1:
                available_node_ids.append(job_and_task_to_node(job_id, task_id, self.n_machines))
        mask = torch.zeros(self.n_nodes ** 2)
        for node_id in available_node_ids:
            mask[node_id + self.n_nodes * node_id] = 1
        return mask
