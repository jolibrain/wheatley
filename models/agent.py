from stable_baselines3.common.callbacks import EveryNTimesteps
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.ppo import PPO
from stable_baselines3.a2c import A2C
import torch

from env.env import Env
from models.agent_callback import TestCallback
from models.policy import Policy
from models.features_extractor import FeaturesExtractor
from problem.problem_description import ProblemDescription

from config import DEVICE


class Agent:
    def __init__(
        self,
        n_epochs=None,
        n_steps_episode=None,
        batch_size=None,
        gamma=None,
        clip_range=None,
        ent_coef=None,
        vf_coef=None,
        lr=None,
        optimizer=None,
        add_machine_id=False,
        input_dim_features_extractor=None,
        gconv_type='gin',
        model=None,
    ):
        fake_env = Agent._create_fake_env(add_machine_id)
        if model is not None:
            self.model = model
            self.model.set_env(fake_env)
        else:
            if optimizer.lower() == "adam":
                optimizer_class = torch.optim.Adam
            elif optimizer.lower() == "sgd":
                optimizer_class = torch.optim.SGD
            else:
                raise Exception("Optimizer not recognized")

            self.model = PPO(
                Policy,
                fake_env,
                n_epochs=n_epochs,
                n_steps=n_steps_episode,
                batch_size=batch_size,
                gamma=gamma,
                learning_rate=lr,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                verbose=2,
                policy_kwargs={
                    "features_extractor_class": FeaturesExtractor,
                    "features_extractor_kwargs": {"input_dim_features_extractor": input_dim_features_extractor,
                                                  "gconv_type": gconv_type},
                    "optimizer_class": optimizer_class,
                },
                device=DEVICE,
                gae_lambda=1,  # To use same vanilla advantage function
            )
        self.add_machine_id = add_machine_id

    def save(self, path):
        self.model.save(path)

    @classmethod
    def load(cls, path, add_machine_id):
        return cls(model=PPO.load(path, Agent._create_fake_env(add_machine_id), DEVICE), add_machine_id=add_machine_id)

    def train(
        self,
        problem_description,
        total_timesteps,
        n_test_env,
        eval_freq,
        normalize_input,
        display_env,
        n_workers,
        multiprocessing,
        path,
        fixed_benchmark,
    ):
        # First setup callbacks during training
        test_callback = TestCallback(
            env=Env(problem_description, normalize_input, self.add_machine_id),
            n_test_env=n_test_env,
            display_env=display_env,
            path=path,
            fixed_benchmark=fixed_benchmark,
        )
        event_callback = EveryNTimesteps(n_steps=eval_freq, callback=test_callback)

        # Then launch training
        env_fns = [self._get_env_fn(problem_description, normalize_input) for _ in range(n_workers)]
        vec_env_class = SubprocVecEnv if multiprocessing else DummyVecEnv
        vec_env = vec_env_class(env_fns)
        self.model.set_env(vec_env)
        self.model.learn(total_timesteps, callback=event_callback)

    def predict(self, problem_description):
        env = Env(problem_description, add_machine_id=self.add_machine_id)
        observation = env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)
        solution = env.get_solution()
        return solution

    @staticmethod
    def _create_fake_env(add_machine_id):
        return Env(ProblemDescription(2, 2, 99, "L2D", "L2D"), False, add_machine_id)

    def _get_env_fn(self, problem_description, normalize_input):
        def f():
            return Env(problem_description, normalize_input, self.add_machine_id)

        return f
