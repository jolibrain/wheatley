from env.transition_model import TransitionModel
from utils.utils import job_and_task_to_node


class L2DTransitionModel(TransitionModel):
    def __init__(self, affectations, durations):
        super(L2DTransitionModel, self).__init__(
            affectations, durations, node_encoding="L2D"
        )

    def run(self, action):
        job_id = action
        if job_id >= self.n_jobs:
            return
        task_id = self.state.get_first_unaffected_task(job_id)
        if task_id == -1:  # If all tasks are affected on this job, do nothing
            return
        machine_id = self.affectations[job_id, task_id]
        machine_occupancy = self.state.get_machine_occupancy(machine_id)
        job_availability_time = self.state.get_job_availability(
            job_id, task_id
        )

        node_id = job_and_task_to_node(job_id, task_id, self.n_machines)

        # If no task is affected on machine, just affect it wherever possible and returns
        if not machine_occupancy:
            self.state.affect_node(node_id)
            return

        # Checks wheter task is inserted at the begining, in between or at the end
        if (job_availability_time < machine_occupancy[0][0]):
            # Insert task before all other tasks
            self.state.set_precedency(node_id, machine_occupancy[0][2])
        else:
            # Find where there are free times, and check if we can insert task
            index = -1
            for i in range(len(machine_occupancy) - 1):
                start_time, duration, node_id = machine_occupancy[i]
                next_start_time, next_duration, next_node_id = machine_occupancy[i+1]
                if start_time + duration < next_start_time:
                    if start_time + duration <= job_availability_time and job_availability_time < next_start_time:
                        index = i
            if index == -1:
                # The job can be inserted nowhere, so we add it at the end
                self.state.set_precedency(machine_occupancy[-1][2], node_id)
            else:
                # The job is inserted between task_index and task_index+1
                self.state.set_precedency(machine_occupancy[index][2], node_id)
                self.state.set_precedency(
                    node_id, machine_occupancy[index + 1][2]
                )
        self.state.affect_node(node_id)
