from stable_baselines3.ppo import PPO

from models.policy import Policy
from models.features_extractor import FeaturesExtractor

from config import DEVICE


class Agent:
    def __init__(self, env):
        self.model = PPO(
            Policy,
            env,
            policy_kwargs={"features_extractor_class": FeaturesExtractor},
            device=DEVICE,
        )
