import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch

from env import Env
from models.policy import Policy
from models.features_extractor import FeaturesExtractor
from config import args, MAX_N_JOBS, MAX_N_MACHINES


def main():

    if args.n_j > MAX_N_JOBS or args.n_m > MAX_N_MACHINES:
        raise Exception("MAX_N_JOBS or MAX_N_MACHINES are too low for this setup")

    training_env = Env(n_jobs=args.n_j, n_machines=args.n_m, n_features=2)
    check_env(training_env)
    model = PPO(
        Policy,
        training_env,
        verbose=2,
        batch_size=4,
        policy_kwargs={"features_extractor_class": FeaturesExtractor},
    )
    model.learn(total_timesteps=25000)

    testing_env = Env(n_jobs=args.n_j_testing, n_machines=n_m_testing, n_features=2)
    obs = testing_env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, rewards, dones, info = testing_env.step(action)


if __name__ == "__main__":
    main()
