import gym
from gym.spaces import Discrete, Box
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo import PPO
import torch

from models.policy import Policy

from config import HIDDEN_DIM_FEATURES_EXTRACTOR, MAX_N_EDGES, MAX_N_NODES


def test_ppo():
    env = Env()
    ppo = PPO(
        Policy,
        env,
        n_steps=2048,
        batch_size=64,
        learning_rate=2e-5,
        policy_kwargs={"features_extractor_class": DumbFeaturesExtractor},
    )
    ppo.learn(total_timesteps=5000)
    print(
        ppo.policy.evaluate_actions(
            torch.tensor([env.get_observation() for _ in range(16)]),
            torch.tensor(
                [
                    0,
                    1,
                    2,
                    3,
                    MAX_N_NODES,
                    MAX_N_NODES + 1,
                    MAX_N_NODES + 2,
                    MAX_N_NODES + 3,
                    2 * MAX_N_NODES,
                    2 * MAX_N_NODES + 1,
                    2 * MAX_N_NODES + 2,
                    2 * MAX_N_NODES + 3,
                    3 * MAX_N_NODES,
                    3 * MAX_N_NODES + 1,
                    3 * MAX_N_NODES + 2,
                    3 * MAX_N_NODES + 3,
                ]
            ),
        )[1]
    )
    env.reset()
    assert ppo.predict(env.get_observation(), deterministic=True)[0] == 0
    switched_observation = env.get_observation()[[0, 2, 1, 3, 4]]
    assert (
        ppo.predict(switched_observation, deterministic=True)[0]
        == 1 * MAX_N_NODES + 1
    )


class Env(gym.Env):
    def __init__(self):
        self.action_space = Discrete(MAX_N_EDGES + 1)
        self.observation_space = Box(
            0, 6, (5, HIDDEN_DIM_FEATURES_EXTRACTOR + 4)
        )
        self.counter = 0

    def step(self, action):
        self.counter += 1
        if action == 0:
            reward = 1
        else:
            reward = 0
        next_observation = self.get_observation()
        info = {}
        done = self.done()
        return next_observation, reward, done, info

    def reset(self):
        self.counter = 0
        observation = self.get_observation()
        return observation

    def get_observation(self):
        return np.array(
            [
                [1 for i in range(HIDDEN_DIM_FEATURES_EXTRACTOR)]
                + [0 for i in range(4)],
                [2 for i in range(HIDDEN_DIM_FEATURES_EXTRACTOR)]
                + [1 for i in range(4)],
                [3 for i in range(HIDDEN_DIM_FEATURES_EXTRACTOR)]
                + [1 for i in range(4)],
                [4 for i in range(HIDDEN_DIM_FEATURES_EXTRACTOR)]
                + [1 for i in range(4)],
                [5 for i in range(HIDDEN_DIM_FEATURES_EXTRACTOR)]
                + [1 for i in range(4)],
            ]
        )

    def done(self):
        return self.counter >= 30


class DumbFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(DumbFeaturesExtractor, self).__init__(
            observation_space=observation_space,
            features_dim=(HIDDEN_DIM_FEATURES_EXTRACTOR + 4) * 5,
        )

    def forward(self, obs):
        return obs
