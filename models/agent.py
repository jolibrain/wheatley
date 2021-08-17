from stable_baselines3.ppo import PPO

from models.policy import Policy
from models.features_extractor import FeaturesExtractor

from config import DEVICE


class Agent:
    def __init__(self, env, model=None):
        if model is not None:
            self.model = model
            self.model.set_env(env)
        else:
            self.model = PPO(
                Policy,
                env,
                policy_kwargs={"features_extractor_class": FeaturesExtractor},
                device=DEVICE,
            )

    @classmethod
    def load(cls, path, env):
        return cls(env, PPO.load(path, env, DEVICE))

    def save(self, path):
        self.model.save(path)

    def train(self, problem_description, n_steps):
        pass

    def predict(self, problem_description):
        pass
