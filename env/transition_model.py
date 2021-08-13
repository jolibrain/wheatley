from abc import ABC, abstractmethod


class TransitionModel(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()

    @abstractmethod
    def get_graph(self):
        raise NotImplementedError()

    @abstractmethod
    def done(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()
