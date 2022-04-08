from abc import ABC, abstractmethod


class TransitionModel(ABC):
    def __init__(self, affectations, durations, max_n_jobs, max_n_machines):
        pass

    @abstractmethod
    def run(self, job_id):
        pass

    @abstractmethod
    def get_mask(self, state):
        pass
