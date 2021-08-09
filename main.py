import gym
from stable_baselines3 import PPO

from env.env import Env
from models.policy import Policy

def main():
    training_env = Env(n_jobs=5, n_machines=5, n_features=2)
    model = PPO(Policy, training_env, verbose=1)
    model.learn(total_timesteps=25000)
    
    testing_env = Env(n_jobs=8, n_machines=8, n_features=2)
    obs = testing_env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, rewards, dones, info = testing_env.step(action)


if __name__ == "__main__":
    main()
