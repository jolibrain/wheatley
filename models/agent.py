from stable_baselines3.ppo import PPO

from env.env import Env
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

    def save(self, path):
        self.model.save(path)

    @classmethod
    def load(cls, path, env):
        return cls(env, PPO.load(path, env, DEVICE))

    def train(self, problem_description, total_timesteps):
        env = Env(problem_description)
        self.model.set_env(env)
        self.model.learn(total_timesteps)

    def predict(self, problem_description):
        env = Env(problem_description)
        self.model.set_env(env)
        observation = env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(observation)
            observation, reward, done, info = env.step(action)
        solution = env.get_solution()
        return solution
