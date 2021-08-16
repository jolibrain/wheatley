import numpy as np
from torch_geometric.data import Data
import torch

from env.transition_model import TransitionModel


class L2DTransitionModel(TransitionModel):
    def __init__(self, n_jobs, n_machines, affectations, durations):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.affectations = affectations
        self.durations = durations

        # An array to store the starting time of each task. These starting times are
        # lower bound estimates, since they can grow during execution
        self.task_starting_time = None
        self.machine_occupancy = None
        # An array to tell, for each job, how many tasks of this job were assigned
        self.number_assigned_tasks = None

        self.graph = None

        self.reset()

    def step(self, action):
        """
        Unlike the step function of an env, this function doesn't return any state. The
        state is indeed computed in the get_graph function
        """
        # Filled up actions don't make the transition model evolve
        if action > self.n_jobs:
            return
        job = action
        task_rank = self.number_assigned_tasks[job]
        machine = self.affectations[job, task_rank]
        duration = self.durations[job, task_rank]

        if task_rank == self.n_machines:
            return  # In the case there are no more tasks in this job, do nothing

        # First compute the time at which the preceding task will be done
        if task_rank == 0:
            job_ready_time = 0
        else:
            job_ready_time = (
                self.task_starting_times[job, task_rank - 1]
                + self.durations[job, task_rank - 1]
            )

        # Then compute the times at which the machine is available for the task
        machine_availability = self.machine_occupancy.get_machine_availability(
            machine
        )

        # Then take the first time at which both machine is available and job is ready :
        # this will be the starting time of the considered job
        start_time = -1
        for (
            availability_start_time,
            availability_duration,
        ) in machine_availability:
            if availability_start_time >= job_ready_time:
                start_time = availability_start_time
                break
        if start_time == -1:  # This means job_ready_time is the bottleneck
            start_time = job_ready_time

        # And now that we choose the starting_time for our job, we need to insert it in
        # the timeline
        self.machine_occupancy.insert_task(
            machine, task_rank, job, start_time, duration
        )
        self.tasks_starting_times[job, task_rank] = start_time
        self.number_assigned_tasks[job] += 1

        # Finally, we update the graph of precedency
        # TODO

    def get_graph(self):
        return self.graph

    def done(self):
        """
        The env is done when all jobs have all their tasks assigned
        """
        done = int(np.min(self.number_assigned_tasks)) == self.n_machines
        return done

    def reset(self):
        self.task_starting_times = -1 * np.ones_like(self.affectations)
        self.machine_occupancy = MachineOccupancy(
            self.n_machines, self.affectations
        )
        self.number_assigned_tasks = np.zeros(self.n_jobs)

        # Define initial graph
        edges_first_node = np.concatenate(
            [
                [
                    job_index * self.n_machines + i
                    for i in range(self.n_machines - 1)
                ]
                for job_index in range(self.n_jobs)
            ]
        )
        edges_second_node = edges_first_node + 1
        edge_index = np.vstack((edges_first_node, edges_second_node))

        scheduled = np.zeros(self.n_jobs * self.n_machines)
        lower_bounds = np.cumsum(self.durations, axis=1).flatten()
        features = np.vstack((scheduled, lower_bounds)).transpose()

        self.graph = Data(
            x=torch.tensor(features),
            edge_index=torch.tensor(edge_index, dtype=torch.int64),
        )


class MachineOccupancy:
    def __init__(self, n_machines, affectations):
        self.n_machines = n_machines
        self.affectations = affectations
        # The machine occupancy tab stores all tasks that are assigned to the machine
        # with their starting time and duration. Note : the tasks should always be
        # ordered by increasing starting time
        self.machine_tasks = [[] for _ in range(self.n_machines)]

    def get_machine_availability(self, machine_index):
        """
        Return the availability of the wanted machine under the form of tab of tuples
        (availability_start_time, availability_duration). The last available period has
        a duration of -1 to represent infinity.
        """
        availability = [(0, -1)]
        for task in self.machine_tasks[machine_index]:
            (task_start_time, task_duration, _, _) = task
            if task_start_time == availability[-1][0]:
                availability[-1][0] += task_duration
            else:
                availability[-1][1] = task_start_time - availability[-1][0]
                availability.append((task_start_time + task_duration, -1))
        return availability

    def insert_task(self, machine, job, task_rank, start_time, duration):
        conflicting_task_ids = self.check_conflicts(
            machine, start_time, duration
        )

        if conflicting_task_ids:
            # We solve conflicts, beginning with the last ones, so they enter in
            # conflict with the first ones, and get resolved a good way
            for task_id in reversed(conflicting_task_ids):
                task_start_time, _, _, _ = self.machine_tasks[machine][task_id]
                task_shift = start_time + duration - task_start_time
                self.shift_task(machine, task_id, task_shift)

        # And finally insert new task at the right place
        new_task = (start_time, duration, job, task_rank)
        index = -1
        for i, task in self.machine_tasks[machine]:
            (task_start_time, _, _, _) = task
            if task_start_time > start_time:
                index = i
                break
        self.machine_tasks[machine].insert(index, new_task)

    def check_conflict(self, machine, start_time, duration):
        """
        Check wether it is possible to insert a task at starting_time with duration time
        on a machine. returns False if possible, and returns conflicting tasks ids if
        impossible.
        """
        conflicting_tasks_ids = []
        for i, task in enumerate(self.machine_tasks[machine]):
            (task_start_time, task_duration, _, _) = task
            conflict = (
                True
                if (
                    (task_start_time + task_duration < start_time + duration)
                    and (task_start_time + task_duration > start_time)
                )
                or (
                    (task_start_time > start_time)
                    and (task_start_time < start_time + duration)
                )
                or (
                    (start_time > task_start_time)
                    and (start_time < task_start_time + task_duration)
                )
                or (
                    (start_time + duration > task_start_time)
                    and (start_time + duration < task_start_time + duration)
                )
                else False
            )

            if conflict:
                conflicting_tasks_ids.append(i)

        return (
            conflicting_tasks_ids if len(conflicting_tasks_ids) != 0 else False
        )

    def shift_task(self, machine, task_id, task_shift):
        """
        Shift a task by a certain time, and take care of solving all inherent conflicts
        by recursively shifting all conflicting tasks
        """
        task = self.machine_tasks[machine][task_id]
        (task_start_time, task_duration, job, task_rank) = task
        conflicting_task_ids = self.check_conflicts(
            machine, task_start_time + task_shift, task_duration
        )
        for i in reversed(conflicting_task_ids):
            cur_task_start_time, _, _, _ = self.machine_occupancy[machine][i]
            cur_task_shift = (
                task_start_time
                + task_shift
                + task_duration
                - cur_task_start_time
            )
            self.shift_task(machine, i, cur_task_shift)

        next_job_task_id, next_job_machine = self.get_next_job_task(
            job, task_rank
        )
        if next_job_task_id != -1:
            next_job_start_time = self.machine_tasks[next_job_machine][
                next_job_task_id
            ][0]
            if (
                next_job_start_time
                < task_start_time + task_shift + task_duration
            ):
                self.shift_task(
                    next_job_machine,
                    next_job_task_id,
                    task_start_time
                    + task_shift
                    + task_duration
                    - next_job_start_time,
                )

    def get_next_job_task_id(self, job, task_rank):
        """
        This function returns the task_id (in the list of machine occupancy) of the next
        task of the job
        """
        task_id = -1
        machine = self.affectations[job, task_rank]
        for cur_task_id, cur_task in enumerate(self.machine_tasks[machine]):
            cur_job, cur_task_rank = cur_task[2], cur_task[3]
            if cur_job == job and cur_task_rank == task_rank:
                task_id = cur_task_id
                break
        return task_id, machine
