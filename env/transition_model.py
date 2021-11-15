from abc import ABC, abstractmethod

from env.state import State


class TransitionModel(ABC):
    def __init__(self, affectations, durations, max_n_jobs, max_n_machines):
        self.affectations = affectations
        self.durations = durations

        self.n_jobs = self.affectations.shape[0]
        self.n_machines = self.affectations.shape[1]
        self.n_nodes = self.n_jobs * self.n_machines

        self.max_n_jobs = max_n_jobs
        self.max_n_machines = max_n_machines

        self.state = State(self.affectations, self.durations, self.max_n_jobs, self.max_n_machines)

    @abstractmethod
    def run(self, job_id):
        pass

    @abstractmethod
    def get_mask(self):
        pass

    def get_graph(self, normalize_input, input_list):
        return self.state.to_torch_geometric(normalize_input, input_list)

    def done(self):
        return self.state.done()

    def reset(self):
        self.state.reset()

    def is_uncertain(self):
        return self.state.durations.shape[2] > 1
