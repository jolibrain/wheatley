from abc import ABC, abstractmethod

from env.state import State


class TransitionModel(ABC):
    def __init__(self, affectations, durations, node_encoding):
        self.affectations = affectations
        self.durations = durations
        self.node_encoding = node_encoding

        self.n_jobs = self.affectations.shape[0]
        self.n_machines = self.affectations.shape[1]
        self.n_nodes = self.n_jobs * self.n_machines

        self.state = State(self.affectations, self.durations, node_encoding)

    @abstractmethod
    def run(self, first_node_id, second_node_id):
        pass

    @abstractmethod
    def get_mask(self):
        pass

    def get_graph(self, add_machine_id, normalize_input, one_hot_machine_id):
        return self.state.to_torch_geometric(add_machine_id, normalize_input, one_hot_machine_id)

    def done(self):
        return self.state.done()

    def reset(self):
        self.state.reset()
