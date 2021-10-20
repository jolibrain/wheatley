import torch

from env.transition_model import TransitionModel
from utils.utils import job_and_task_to_node, node_to_job_and_task


class L2DTransitionModel(TransitionModel):
    def __init__(self, affectations, durations, node_encoding, slot_locking):
        super(L2DTransitionModel, self).__init__(affectations, durations, node_encoding)
        self.useless_timesteps = 0
        self.slot_locking = slot_locking
        if self.slot_locking:
            self.slot_availability = [[] for _ in range(affectations.shape[1])]

    def run(self, first_node_id, second_node_id, force_insert, slot_lock):  # noqa
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
            if self.slot_locking:
                self.slot_availability[machine_id].append(0 if slot_lock else 1)
            return

        # Use slot locking (forced insertions are sometimes available)
        if self.slot_locking:
            # Checks wheter task is inserted at the begining, in between or at the end
            if job_availability_time < machine_occupancy[0][0] and (
                not self.slot_availability[machine_id] or (self.slot_availability[machine_id][0] == 1)
            ):
                # Insert task before all other tasks
                self.state.set_precedency(node_id, machine_occupancy[0][2])
                self.slot_availability[machine_id].insert(0, 0 if slot_lock else 1)

            else:
                # Find where there are free times, and check if we can insert task
                index = -1
                for i in range(len(machine_occupancy) - 1):
                    start_time, duration, _ = machine_occupancy[i]
                    next_start_time, next_duration, _ = machine_occupancy[i + 1]
                    if start_time + duration < next_start_time:
                        if job_availability_time < next_start_time:
                            if self.slot_availability[machine_id][i + 1] == 1:
                                index = i
                                break
                if index == -1:
                    # The job can be inserted nowhere, so we add it at the end
                    self.state.set_precedency(machine_occupancy[-1][2], node_id)
                    self.slot_availability[machine_id].append(0 if slot_lock else 1)
                else:
                    # The job is inserted between task_index and task_index+1
                    self.state.remove_precedency(machine_occupancy[index][2], machine_occupancy[index + 1][2])
                    self.state.set_precedency(machine_occupancy[index][2], node_id)
                    self.state.set_precedency(node_id, machine_occupancy[index + 1][2])
                    self.slot_availability[machine_id].insert(index + 1, 0 if slot_lock else 1)

        # If forced insertion are not allowed at all
        elif not force_insert:
            job_duration = self.durations[job_id, task_id]
            # Checks wheter task is inserted at the begining, in between or at the end
            if job_availability_time + job_duration <= machine_occupancy[0][0]:
                # Insert task before all other tasks
                self.state.set_precedency(node_id, machine_occupancy[0][2])

            else:
                # Find where there are free times, and check if we can insert task
                index = -1
                for i in range(len(machine_occupancy) - 1):
                    start_time, duration, _ = machine_occupancy[i]
                    next_start_time, next_duration, _ = machine_occupancy[i + 1]
                    if start_time + duration < next_start_time:
                        if job_duration <= next_start_time - max(start_time + duration, job_availability_time):
                            index = i
                            break
                if index == -1:
                    # The job can be inserted nowhere, so we add it at the end
                    self.state.set_precedency(machine_occupancy[-1][2], node_id)
                else:
                    # The job is inserted between task_index and task_index+1
                    self.state.remove_precedency(machine_occupancy[index][2], machine_occupancy[index + 1][2])
                    self.state.set_precedency(machine_occupancy[index][2], node_id)
                    self.state.set_precedency(node_id, machine_occupancy[index + 1][2])

        # If forced insertion is allowed and performed
        else:
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
                        if job_availability_time < next_start_time:
                            index = i
                            break
                if index == -1:
                    # The job can be inserted nowhere, so we add it at the end
                    self.state.set_precedency(machine_occupancy[-1][2], node_id)
                else:
                    # The job is inserted between task_index and task_index+1
                    self.state.remove_precedency(machine_occupancy[index][2], machine_occupancy[index + 1][2])
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
