from copy import deepcopy
import time

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
import visdom

from env.env import Env
from models.random_agent import RandomAgent
from problem.problem_description import ProblemDescription
from utils.utils_testing import get_ortools_makespan
from utils.utils import load_benchmark


class TestCallback(BaseCallback):
    def __init__(self, env, n_test_env, display_env, path, fixed_benchmark, verbose=2):
        super(TestCallback, self).__init__(verbose=verbose)
        self.testing_env = env
        self.n_test_env = n_test_env
        self.vis = visdom.Visdom(env=display_env)
        self.path = path
        self.fixed_benchmark = fixed_benchmark

        self.n_jobs = self.testing_env.n_jobs
        self.n_machines = self.testing_env.n_machines
        self.max_duration = self.testing_env.max_duration
        self.transition_model_config = self.testing_env.transition_model_config
        self.reward_model_config = self.testing_env.reward_model_config

        self.random_agent = RandomAgent()

        if self.fixed_benchmark:
            self._init_testing_envs()
            self.n_test_env = 100
            self.testing_env = None
        else:
            self.testing_envs = [deepcopy(self.testing_env) for _ in range(self.n_test_env)]
            self.testing_env = None

        self.makespans = []
        self.ortools_makespans = []
        self.random_makespans = []
        self.entropy_losses = []
        self.policy_gradient_losses = []
        self.value_losses = []
        self.losses = []
        self.approx_kls = []
        self.clip_fractions = []
        self.explained_variances = []
        self.clip_ranges = []
        self.ep_len_means = []
        self.ep_rew_means = []
        self.fpss = []
        self.total_timestepss = []
        self.first_callback = True
        self.figure = None

    def _init_testing_envs(self):
        data = load_benchmark(self.n_jobs, self.n_machines)
        self.testing_envs = [
            Env(
                ProblemDescription(
                    self.n_jobs,
                    self.n_machines,
                    self.max_duration,
                    self.transition_model_config,
                    self.reward_model_config,
                    data[i][0],
                    data[i][1],
                ),
                divide_loss=self.testing_env.divide_loss,
                add_machine_id=self.testing_env.add_machine_id,
            )
            for i in range(data.shape[0])
        ]

    def _on_step(self):
        self._evaluate_agent()
        self._save_if_best_model()
        self._visdom_metrics()
        print(self.path)
        return True

    def _save_if_best_model(self):
        min_ratio = np.min(np.array(self.makespans) / np.array(self.ortools_makespans))
        if self.makespans[-1] / self.ortools_makespans[-1] == min_ratio:
            self.model.save(self.path)

    def _evaluate_agent(self):
        mean_makespan = 0
        ortools_mean_makespan = 0
        random_mean_makespan = 0
        for i in range(self.n_test_env):
            obs = self.testing_envs[i].reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.testing_envs[i].step(action)
            schedule = self.testing_envs[i].get_solution().schedule
            durations = self.testing_envs[i].durations
            mean_makespan += np.max(schedule + durations) / self.n_test_env

            ortools_mean_makespan += (
                get_ortools_makespan(
                    self.n_jobs,
                    self.n_machines,
                    self.max_duration,
                    self.testing_envs[i].affectations,
                    self.testing_envs[i].durations,
                )
                / self.n_test_env
            )

            random_mean_makespan += (
                np.max(
                    self.random_agent.predict(
                        ProblemDescription(
                            self.testing_envs[i].n_jobs,
                            self.testing_envs[i].n_machines,
                            self.testing_envs[i].max_duration,
                            self.testing_envs[i].transition_model_config,
                            self.testing_envs[i].reward_model_config,
                            self.testing_envs[i].affectations,
                            self.testing_envs[i].durations,
                        )
                    ).schedule
                    + self.testing_envs[i].durations
                )
                / self.n_test_env
            )

        self.makespans.append(mean_makespan)
        self.ortools_makespans.append(ortools_mean_makespan)
        self.random_makespans.append(random_mean_makespan)

    def _visdom_metrics(self):
        self.vis.line(
            Y=np.array([self.makespans, self.random_makespans, self.ortools_makespans]).T,
            win="test_makespan",
            opts={
                "legend": ["PPO", "Random", "OR-tools"],
                "linecolor": np.array([[31, 119, 180], [255, 127, 14], [44, 160, 44]]),
            },
        )
        self.vis.line(
            Y=np.stack(
                [
                    np.array(self.makespans) / np.array(self.ortools_makespans),
                    np.array(self.random_makespans) / np.array(self.ortools_makespans),
                ]
            ).T,
            win="test_makespan_ratio",
            opts={
                "legend": ["PPO / OR-tools", "Random / OR-tools"],
                "linecolor": np.array([[31, 119, 180], [255, 127, 14]]),
            },
        )

        if self.first_callback:
            self.first_callback = False
            return

        self.entropy_losses.append(self.model.ent_coef * self.model.logger.name_to_value["train/entropy_loss"])
        self.policy_gradient_losses.append(self.model.logger.name_to_value["train/policy_gradient_loss"])
        self.value_losses.append(self.model.vf_coef * self.model.logger.name_to_value["train/value_loss"])
        self.losses.append(self.model.logger.name_to_value["train/loss"])
        self.approx_kls.append(self.model.logger.name_to_value["train/approx_kl"])
        self.clip_fractions.append(self.model.logger.name_to_value["train/clip_fraction"])
        self.explained_variances.append(self.model.logger.name_to_value["train/explained_variance"])
        self.clip_ranges.append(self.model.logger.name_to_value["train/clip_range"])
        # Recreate last features by hand, since they are erased
        self.ep_rew_means.append(safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))
        self.ep_len_means.append(safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]))
        self.fpss.append(int(self.model.num_timesteps / (time.time() - self.model.start_time)))
        self.total_timestepss.append(self.model.num_timesteps)

        figure, ax = plt.subplots(3, 4, figsize=(16, 12))

        ax[0, 0].plot(self.entropy_losses)
        ax[0, 0].set_title("entropy_loss")
        ax[0, 1].plot(self.policy_gradient_losses)
        ax[0, 1].set_title("policy_gradient_loss")
        ax[0, 2].plot(self.value_losses)
        ax[0, 2].set_title("value_loss")
        ax[0, 3].plot(self.losses)
        ax[0, 3].set_title("loss")
        ax[1, 0].plot(self.approx_kls)
        ax[1, 0].set_title("approx_kl")
        ax[1, 1].plot(self.clip_fractions)
        ax[1, 1].set_title("clip_fraction")
        ax[1, 2].plot(self.explained_variances)
        ax[1, 2].set_title("explained_variance")
        ax[1, 3].plot(self.clip_ranges)
        ax[1, 3].set_title("clip_range")
        ax[2, 0].plot(self.ep_len_means)
        ax[2, 0].set_title("ep_len_mean")
        ax[2, 1].plot(self.ep_rew_means)
        ax[2, 1].set_title("ep_rew_mean")
        ax[2, 2].plot(self.fpss)
        ax[2, 2].set_title("fps")
        ax[2, 3].plot(self.total_timestepss)
        ax[2, 3].set_title("total_timesteps")

        self.vis.matplot(figure, win="training")

        plt.close(self.figure)
        self.figure = figure
