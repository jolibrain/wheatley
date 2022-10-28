import torch

from env.transition_model import TransitionModel
from utils.utils import job_and_task_to_node, node_to_job_and_task


class SimpleTransitionModel(TransitionModel):
    def __init__(
        self,
        affectations,
        durations,
        max_n_jobs,
        max_n_machines,
        observe_real_duration_when_affect=True,
        stochastic_metric="pessimistic",
    ):
        super(SimpleTransitionModel, self).__init__(affectations, durations, max_n_jobs, max_n_machines)
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

        # Observe duration (for the uncertainty case)
        if self.observe_real_duration_when_affect:
            state.observe_real_duration(node_id, do_update=False)

        if machine_occupancy:
            state.set_precedency(machine_occupancy[-1][2], node_id, do_update=False)
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
