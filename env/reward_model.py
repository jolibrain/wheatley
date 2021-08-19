from abc import ABC, abstractmethod


class RewardModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, state, action, next_state):
        pass
