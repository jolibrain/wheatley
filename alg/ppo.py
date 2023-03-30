#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#
# largely inspired from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .logger import Logger, configure_logger
from collections import deque
from utils.utils import obs_as_tensor, safe_mean, rebatch_obs, get_obs, decode_mask
import tqdm


def create_env(env_cls, problem_description, env_specification, i):
    def _init():
        env = env_cls(problem_description, env_specification, i, validate=False)
        return env

    return _init


class PPO:
    def __init__(
        self,
        agent_specification,
        env_cls,
        validator=None,
    ):

        self.optimizer_class = agent_specification.optimizer_class
        self.logger = configure_logger()
        self.env_cls = env_cls

        self.num_envs = agent_specification.n_workers
        self.gamma = agent_specification.gamma
        self.update_epochs = agent_specification.n_epochs
        self.norm_adv = agent_specification.normalize_advantage
        self.ent_coef = agent_specification.ent_coef
        self.num_steps = agent_specification.n_steps_episode
        self.gae_lambda = 1.0
        self.clip_vloss = False
        self.clip_coef = agent_specification.clip_range
        self.ent_coef = agent_specification.ent_coef
        self.vf_coef = agent_specification.vf_coef
        self.target_kl = agent_specification.target_kl
        self.max_grad_norm = 0.5  # SB3 default
        self.minibatch_size = agent_specification.batch_size
        self.validator = validator

        # in case of resume
        self._num_timesteps_at_start = 0
        self.ep_info_buffer = deque(maxlen=100)

    def collect_rollouts(self, agent, envs, env_specification, data_device):
        # ALGO Logic: Storage setup
        obs = []
        actions = torch.empty(
            (self.num_steps, self.num_envs) + envs.single_action_space.shape
        ).to(data_device)
        logprobs = torch.empty((self.num_steps, self.num_envs)).to(data_device)
        rewards = torch.empty((self.num_steps, self.num_envs)).to(data_device)
        dones = torch.empty((self.num_steps, self.num_envs)).to(data_device)
        values = torch.empty((self.num_steps, self.num_envs)).to(data_device)
        action_masks = torch.empty(
            (self.num_steps, self.num_envs, env_specification.max_n_nodes)
        ).to(data_device)

        # buffer filling
        o, info = envs.reset()
        # next obs is a list of dicts
        next_obs = obs_as_tensor(o)
        action_mask = decode_mask(info["mask"])
        next_done = torch.empty(self.num_envs).to(data_device)

        self.ep_info_buffer = deque(maxlen=100)
        self.global_step += self.num_envs * self.num_steps

        for step in tqdm.tqdm(range(0, self.num_steps), desc="   collecting rollouts"):
            obs.append(next_obs)
            action_masks[step] = torch.tensor(action_mask)
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs, action_masks=action_mask
                )

            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
            action_mask = decode_mask(info["mask"])
            if "final_info" in info:
                self.ep_info_buffer.extend(
                    [ep_info["episode"] for ep_info in info["final_info"]]
                )

            next_obs = obs_as_tensor(next_obs)
            rewards[step] = torch.tensor(reward).view(-1).to(data_device)
            next_done = torch.Tensor(done).to(data_device)

        # compute returns and advantages
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1).to(data_device)
            advantages = torch.empty_like(rewards)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = (
                    rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = rebatch_obs(obs)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_action_masks = action_masks.reshape(-1, env_specification.max_n_nodes)

        return (
            b_obs,
            b_logprobs,
            b_actions,
            b_advantages,
            b_returns,
            b_values,
            b_action_masks,
        )

    def train(
        self,
        agent,
        problem_description,
        env_specification,
        training_specification,
        lr,
        log_interval=1,
        rollout_data_device=torch.device("cpu"),
        rollout_agent_device=torch.device("cpu"),
        train_device=torch.device("cpu"),
        opt_state_dict=None,
        skip_initial_eval=False,
    ):
        # env setup
        batch_size = self.num_envs * self.num_steps
        classVecEnv = gym.vector.AsyncVectorEnv
        print("creating environments")
        if training_specification.vecenv_type == "dummy":
            envs = gym.vector.SyncVectorEnv(
                [
                    create_env(self.env_cls, problem_description, env_specification, i)
                    for i in range(self.num_envs)
                ],
            )
        else:
            envs = gym.vector.AsyncVectorEnv(
                [
                    create_env(self.env_cls, problem_description, env_specification, i)
                    for i in range(self.num_envs)
                ],
                # spwan helps when observation space is huge
                # context="spawn",
                copy=False,
            )

        print("... done creating environments")

        self.optimizer = self.optimizer_class(
            agent.parameters(),
            lr=lr,
        )
        if opt_state_dict is not None:
            self.optimizer.load_state_dict(opt_state_dict)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        print("optimizer", self.optimizer)
        print("collecting rollouts using", rollout_agent_device)
        print("storing rollouts on", rollout_data_device)
        print("learning on", train_device)

        self.global_step = 0
        if not skip_initial_eval:
            print("initial validation")
            agent.to(rollout_agent_device)
            self.validator.validate(agent, self)
            print("... done initial validation")
        start_time = time.time()
        num_updates = training_specification.total_timesteps // batch_size

        self.n_epochs = 0
        self.start_time = time.time()
        for update in range(1, num_updates + 1):

            print("UPDATE ", update)

            agent.to(rollout_agent_device)
            # collect data with current policy
            (
                b_obs,
                b_logprobs,
                b_actions,
                b_advantages,
                b_returns,
                b_values,
                b_action_masks,
            ) = self.collect_rollouts(
                agent,
                envs,
                env_specification,
                rollout_data_device,
            )

            # Optimizing the policy and value network
            b_inds = np.arange(batch_size)
            clipfracs = []

            entropy_losses = []
            pg_losses = []
            value_losses = []
            approx_kl_divs = []

            agent.to(train_device)
            for epoch in tqdm.tqdm(
                range(self.update_epochs), desc="   epochs             "
            ):
                self.n_epochs += 1
                np.random.shuffle(b_inds)
                approx_kl_divs_on_epoch = []
                for start in tqdm.tqdm(
                    range(0, batch_size, self.minibatch_size),
                    desc="   minibatches        ",
                    leave=False,
                ):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        get_obs(b_obs, mb_inds),
                        action=b_actions.long()[mb_inds].to(train_device),
                        action_masks=b_action_masks[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds].to(train_device)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        approx_kl = ((ratio - 1) - logratio).mean()
                        approx_kl_divs.append(approx_kl.item())
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    if self.target_kl is not None:
                        approx_kl_divs_on_epoch.append(approx_kl.item())

                    mb_advantages = b_advantages[mb_inds].to(train_device)
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = torch.nn.functional.mse_loss(
                            newvalue, b_returns[mb_inds].to(train_device)
                        )
                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    )

                    value_losses.append(v_loss.item())
                    pg_losses.append(pg_loss.item())
                    entropy_losses.append(entropy_loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None:
                    if np.mean(approx_kl_divs_on_epoch) > self.target_kl:
                        print(
                            "stopping update due to too high kl divergence after epoch",
                            epoch,
                            " / ",
                            self.update_epochs,
                        )

                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            iteration = update + self._num_timesteps_at_start
            if log_interval is not None and iteration % log_interval == 0:
                self.logger.record("train/entropy_loss", np.mean(entropy_losses))
                self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
                self.logger.record("train/value_loss", np.mean(value_losses))
                self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
                self.logger.record("train/clip_fraction", np.mean(clipfracs))
                self.logger.record("train/loss", loss.item())
                self.logger.record("train/explained_variance", explained_var)
                self.logger.record(
                    "train/n_epochs",
                    self.n_epochs,
                    exclude="tensorboard",
                )
                self.logger.record("train/clip_range", self.clip_coef)
                fps = int(self.global_step / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record(
                        "rollout/ep_rew_mean",
                        safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                    )
                    self.logger.record(
                        "rollout/ep_len_mean",
                        safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                    )
                self.logger.record("time/fps", fps)
                self.logger.record(
                    "time/dps",
                    int(
                        self.n_epochs
                        * self.num_envs
                        * self.num_steps
                        / (time.time() - self.start_time)
                    ),
                )
                self.logger.record(
                    "time/time_elapsed",
                    int(time.time() - self.start_time),
                    exclude="tensorboard",
                )
                self.logger.record(
                    "time/total_timesteps", self.global_step, exclude="tensorboard"
                )

            if (
                training_specification.validation_freq is not None
                and iteration % training_specification.validation_freq == 0
                and self.validator is not None
            ):
                self.validator.validate(agent, self)
            self.logger.dump(step=self.global_step)

        envs.close()
