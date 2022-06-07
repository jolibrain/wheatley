import torch

from env.transition_model import TransitionModel
from utils.utils import job_and_task_to_node, node_to_job_and_task


class L2DTransitionModel(TransitionModel):
    def __init__(
        self,
        affectations,
        durations,
        max_n_jobs,
        max_n_machines,
        observe_real_duration_when_affect=True,
        stochastic_metric="pessimistic",
    ):
        super(L2DTransitionModel, self).__init__(affectations, durations, max_n_jobs, max_n_machines)
        self.useless_timesteps = 0
        self.observe_real_duration_when_affect = observe_real_duration_when_affect
        self.metric = stochastic_metric
        if self.metric == "realistic":
            self.metric_index = 0
        elif self.metric == "optimistic":
            self.metric_index = 1
        elif self.metric == "pessimistic":
            self.metric_index = 2
        elif self.metric == "averagistic":
            self.metric_index = 3
        else:
            raise Exception("Stochastic metric not recognized")

    def run(self, state, node_id, force_insert):  # noqa

        # If the job_id is bigger that max job_id, we don't operate the action
        job_id, task_id = node_to_job_and_task(node_id, state.n_machines)
        if job_id >= state.n_jobs:
            self.useless_timesteps += 1
            return

        if task_id != state.get_first_unaffected_task(job_id):
            self.useless_timesteps += 1
            return

        machine_id = state.affectations[job_id, task_id]
        if machine_id == -1:
            self.useless_timesteps += 1
            return

        machine_occupancy = state.get_machine_occupancy(machine_id, self.metric)
        job_availability_time = state.get_job_availability(job_id, task_id, self.metric)

        # If no task is affected on machine, just affect it wherever possible and returns
        if not machine_occupancy:
            # Observe duration (for the uncertainty case)
            if self.observe_real_duration_when_affect:
                state.observe_real_duration(node_id, do_update=False)
            state.update_completion_times(node_id, rec=False)
            state.affect_node(node_id)
            return

        # If forced insertion are not allowed
        if not force_insert:
            # Observe duration (for the uncertainty case)
            if self.observe_real_duration_when_affect:
                state.observe_real_duration(node_id, do_update=False)

            job_duration = state.durations[job_id, task_id, self.metric_index]
            # Checks wheter task is inserted at the begining, in between or at the end
            if job_availability_time + job_duration <= machine_occupancy[0][0]:
                # Insert task before all other tasks
                state.set_precedency(node_id, machine_occupancy[0][2], do_update=False)
                if self.observe_real_duration_when_affect:
                    state.update_completion_times(node_id)
                else:
                    state.update_completion_times(machine_occupancy[0][2])
                state.affect_node(node_id)

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
                    state.set_precedency(machine_occupancy[-1][2], node_id, do_update=False)
                    state.update_completion_times(node_id)
                    state.affect_node(node_id)
                else:
                    # The job is inserted between task_index and task_index+1
                    state.remove_precedency(machine_occupancy[index][2], machine_occupancy[index + 1][2])
                    state.set_precedency(machine_occupancy[index][2], node_id, do_update=False)
                    state.set_precedency(node_id, machine_occupancy[index + 1][2], do_update=False)
                    state.update_completion_times(node_id)
                    state.affect_node(node_id)

        # If forced insertion is allowed and performed
        elif force_insert:
            if self.observe_real_duration_when_affect:
                state.observe_real_duration(node_id, do_update=False)
            # Checks wheter task is inserted at the begining, in between or at the end
            if job_availability_time < machine_occupancy[0][0]:
                # Insert task before all other tasks
                state.set_precedency(node_id, machine_occupancy[0][2], do_update=False)
                state.update_completion_times(node_id)
                state.affect_node(node_id)

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
                    state.set_precedency(machine_occupancy[-1][2], node_id, do_update=False)
                    state.update_completion_times(node_id)
                    state.affect_node(node_id)
                else:
                    # The job is inserted between task_index and task_index+1
                    state.remove_precedency(machine_occupancy[index][2], machine_occupancy[index + 1][2])
                    state.set_precedency(machine_occupancy[index][2], node_id, do_update=False)
                    state.set_precedency(node_id, machine_occupancy[index + 1][2], do_update=False)
                    state.update_completion_times(node_id)
                    state.affect_node(node_id)

    def get_mask(self, state, add_boolean=False):
        available_node_ids = []
        for job_id in range(state.n_jobs):
            task_id = state.get_first_unaffected_task(job_id)
            if task_id != -1 and state.affectations[job_id, task_id] != -1:
                available_node_ids.append(job_and_task_to_node(job_id, task_id, state.n_machines))
        mask = [False] * state.n_nodes
        for node_id in available_node_ids:
            mask[node_id] = True
        return mask * (2 if add_boolean else 1)
