from abc import ABC, abstractmethod


class RewardModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, obs, action, next_obs):
        pass
