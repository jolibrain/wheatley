from env.transition_model import TransitionModel
from utils.utils import job_and_task_to_node


class L2DTransitionModel(TransitionModel):
    def __init__(self, affectations, durations):
        super(L2DTransitionModel, self).__init__(
            affectations, durations, node_encoding="L2D"
        )

    def run(self, action):
        job_id = action
        task_id = self.state.get_first_unaffected_task(job_id)
        if task_id == -1:  # If all tasks are affected on this job, do nothing
            return
        machine_id = self.affectations[job_id, task_id]
        machine_availability = self.state.get_machine_availability(machine_id)
        job_availability_time = self.state.get_job_availability(
            job_id, task_id
        )

        # Checks wheter task is inserted at the begining, in between or at the end
        node_id = (job_and_task_to_node(job_id, task_id),)
        if (
            machine_availability[0][0] == 0
            and job_availability_time < machine_availability[0][1]
        ):
            # Insert task before all other tasks
            self.state.set_precedency(node_id, machine_availability[0][2])
        elif machine_availability[-1][0] < job_availability_time:
            # Insert task after all other tasks
            self.state.set_precedency(machine_availability[-1][2], node_id)
        else:
            # Find where and then insert task between other tasks
            index = 0
            for i, (start_time, _, _) in enumerate(machine_availability):
                if start_time > job_availability_time:
                    index = i - 1
            self.state.set_precedency(machine_availability[index][2], node_id)
            self.state.set_precedency(
                node_id, machine_availability[index + 1][2]
            )
        self.state.affect_node(node_id)
