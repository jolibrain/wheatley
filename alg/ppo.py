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
from collections import deque
from functools import partial
import os
import dgl
import glob
import pickle

import gymnasium as gym
from psp.env.graphgym.async_vector_env import AsyncGraphVectorEnv
from generic.agent import calc_twohot, symexp, symlog
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
import tqdm
import math

from generic.utils import decode_mask, safe_mean

from .logger import Logger, configure_logger, monotony, stability


def create_env(env_cls, problem_description, env_specification, i):
    def _init():
        env = env_cls(problem_description, env_specification, i, validate=False)
        return env

    return _init


class PPO:
    def __init__(
        self,
        training_specification,
        env_cls,
        validator=None,
        discard_incomplete_trials=True,
    ):
        self.optimizer_class = training_specification.optimizer_class
        self.logger = configure_logger(
            folder=training_specification.path, format_strings=["json"]
        )
        self.env_cls = env_cls

        self.num_envs = training_specification.n_workers
        self.gamma = training_specification.gamma
        self.update_epochs = training_specification.n_epochs
        self.norm_adv = training_specification.normalize_advantage
        self.ent_coef = training_specification.ent_coef
        self.num_steps = training_specification.n_steps_episode
        self.gae_lambda = training_specification.gae_lambda
        self.clip_vloss = False
        self.clip_coef = training_specification.clip_range
        self.ent_coef = training_specification.ent_coef
        self.vf_coef = training_specification.vf_coef
        self.target_kl = training_specification.target_kl
        self.max_grad_norm = 0.5  # SB3 default
        self.minibatch_size = training_specification.batch_size
        self.iter_size = training_specification.iter_size
        self.validator = validator
        self.vecenv_type = training_specification.vecenv_type
        self.total_timesteps = training_specification.total_timesteps
        self.validation_freq = training_specification.validation_freq
        self.return_based_scaling = training_specification.return_based_scaling
        self.obs_on_disk = training_specification.store_rollouts_on_disk
        self.critic_loss = training_specification.critic_loss

        self.discard_incomplete_trials = discard_incomplete_trials

        # in case of resume
        self._num_timesteps_at_start = 0
        self.ep_info_buffer = deque(maxlen=100)

    def keep_only(self, obs, to_keep):
        kobs = {}
        for key in obs:
            kobs[key] = obs[key][to_keep]
        return kobs

    def collect_rollouts(self, agent, envs, env_specification, data_device, sigma=1.0):
        # ALGO Logic: Storage setup
        obs = []
        actions = torch.empty((self.num_steps, self.num_envs)).to(data_device)
        logprobs = torch.empty((self.num_steps, self.num_envs)).to(data_device)
        rewards = torch.empty((self.num_steps, self.num_envs, agent.reward_dim)).to(
            data_device
        )
        dones = torch.empty((self.num_steps, self.num_envs)).to(data_device)
        values = torch.empty((self.num_steps, self.num_envs, agent.reward_dim)).to(
            data_device
        )
        action_masks = list()

        if self.discard_incomplete_trials:
            to_keep = [[] for i in range(self.num_envs)]
            to_keep_candidate = [[] for i in range(self.num_envs)]

        # buffer filling
        o, info = envs.reset()
        # next obs is a list of dicts
        next_obs = agent.obs_as_tensor(o)
        action_mask = decode_mask(info["mask"])
        next_done = torch.zeros(self.num_envs).to(data_device)

        self.ep_info_buffer = deque(maxlen=100)
        self.global_step += self.num_envs * self.num_steps

        if self.obs_on_disk is not None:
            for f in glob.glob(
                self.obs_on_disk + "/wheatley_dgl_" + str(os.getpid()) + "_*.obs"
            ) + glob.glob(
                self.obs_on_disk + "/wheatley_pkl_" + str(os.getpid()) + "_*.obs"
            ):
                os.remove(f)

        for step in tqdm.tqdm(range(0, self.num_steps), desc="   collecting rollouts"):
            if self.obs_on_disk:
                if agent.graphobs:
                    for i, o in enumerate(next_obs):
                        fname = (
                            self.obs_on_disk
                            + "/wheatley_dgl_"
                            + str(os.getpid())
                            + "_"
                            + str(step * self.num_envs + i)
                            + ".obs"
                        )
                        dgl.save_graphs(fname, [o])
                        obs.append(fname)
                else:
                    list_obs = [dict(zip(next_obs, t)) for t in zip(*next_obs.values())]
                    for i, o in enumerate(list_obs):
                        fname = (
                            self.obs_on_disk
                            + "/wheatley_pkl_"
                            + str(os.getpid())
                            + "_"
                            + str(step * self.num_envs + i)
                            + ".obs"
                        )
                        pickle.dump(o, open(fname, "wb"))
                        obs.append(fname)
            else:
                obs.append(next_obs)
            action_masks.append(torch.tensor(action_mask))
            dones[step] = next_done

            if self.discard_incomplete_trials:
                for i in range(self.num_envs):
                    if dones[step][i] == 1:
                        to_keep[i].extend(to_keep_candidate[i])
                        to_keep_candidate[i].clear()
                    to_keep_candidate[i].append(step)

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs, action_masks=action_mask
                )
                value = agent.get_value_from_logits(value)

            values[step] = value.view(-1, agent.reward_dim)
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
            action_mask = decode_mask(info["mask"])
            if "final_info" in info:
                for ep_info in info["final_info"]:
                    if (
                        ep_info is not None
                    ):  # some episode may be finished and other not
                        self.ep_info_buffer.append(ep_info["episode"])
                # self.ep_info_buffer.extend(
                #     [ep_info["episode"] for ep_info in info["final_info"]]
                # )

            next_obs = agent.obs_as_tensor(next_obs)
            rewards[step] = (
                torch.tensor(reward).view(-1, agent.reward_dim).to(data_device)
            )
            next_done = torch.Tensor(done).to(data_device)

        if self.discard_incomplete_trials:
            for i in range(self.num_envs):
                if next_done[i] == 1:
                    to_keep[i].extend(to_keep_candidate[i])

        with torch.no_grad():
            next_value = (
                agent.get_value_from_logits(agent.get_value(next_obs))
                .reshape(-1, agent.reward_dim)
                .to(data_device)
            )

        if sigma is None:
            # compute return-based scaling as 2105.05347
            with torch.no_grad():
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
                        rewards[t]
                        + self.gamma * nextvalues * nextnonterminal.unsqueeze(1)
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.gamma
                        * self.gae_lambda
                        * nextnonterminal.unsqueeze(1)
                        * lastgaelam
                    )
                returns = advantages + values
                n_dones = int(torch.sum(dones).item())
                gamma = torch.tensor(
                    [self.gamma] * (self.num_steps * self.num_envs - n_dones)
                    + [0.0] * n_dones,
                    dtype=torch.float,
                )
                v_gamma = torch.var(gamma, dim=None).item()
                sigma = math.sqrt(
                    torch.var(rewards, dim=None).item()
                    + v_gamma * torch.mean(returns * returns).item()
                )

        # compute returns and advantages
        with torch.no_grad():
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
                    rewards[t]
                    + self.gamma * nextvalues * nextnonterminal.unsqueeze(1)
                    - values[t]
                ) / sigma
                advantages[t] = lastgaelam = (
                    delta
                    + self.gamma
                    * self.gae_lambda
                    * nextnonterminal.unsqueeze(1)
                    * lastgaelam
                )
            returns = advantages + values

        # Pad the action masks
        max_n_nodes = max(mask.shape[1] for mask in action_masks)
        action_masks = [
            torch.concat(
                (
                    mask,
                    torch.zeros(
                        (mask.shape[0], max_n_nodes - mask.shape[1]),
                        dtype=torch.bool,
                        device=data_device,
                    ),
                ),
                dim=1
            )
            for mask in action_masks
        ]
        action_masks = torch.stack(action_masks, dim=0)

        # flatten the batch
        b_obs = agent.rebatch_obs(obs)
        b_logprobs = logprobs.reshape(-1)
        if agent.graphobs:
            b_actions = actions.reshape((-1))
        else:
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1, agent.reward_dim)
        b_returns = returns.reshape(-1, agent.reward_dim)
        b_values = values.reshape(-1, agent.reward_dim)
        b_action_masks = action_masks.reshape(-1, max_n_nodes)

        if self.discard_incomplete_trials:
            to_keep_b = [
                j + i * self.num_steps for i in range(self.num_envs) for j in to_keep[i]
            ]
            if agent.graphobs or self.obs_on_disk:
                bobs_tokeep = list(b_obs[i] for i in to_keep_b)
            else:
                bobs_tokeep = self.keep_only(b_obs, to_keep_b)
            return (
                bobs_tokeep,
                b_logprobs[to_keep_b],
                b_actions[to_keep_b],
                b_advantages[to_keep_b],
                b_returns[to_keep_b],
                b_values[to_keep_b],
                b_action_masks[to_keep_b],
                sigma,
            )
        return (
            b_obs,
            b_logprobs,
            b_actions,
            b_advantages,
            b_returns,
            b_values,
            b_action_masks,
            sigma,
        )

    def pb_ids(self, problem_description):
        if not hasattr(problem_description, "train_psps"):
            return list(range(self.num_envs))  # simple env id
        # for psps, we should return a list per env of list of problems for this env
        return [list(range(len(problem_description.train_psps)))] * self.num_envs

    def train(
        self,
        agent,
        problem_description,
        env_specification,
        lr,
        weight_decay,
        log_interval=1,
        rollout_data_device=torch.device("cpu"),
        rollout_agent_device=torch.device("cpu"),
        train_device=torch.device("cpu"),
        opt_state_dict=None,
        skip_initial_eval=False,
        skip_model_trace=False,
    ) -> float:
        # env setup
        batch_size = self.num_envs * self.num_steps
        classVecEnv = gym.vector.AsyncVectorEnv
        print("creating environments")
        pbs_per_env = self.pb_ids(problem_description)
        if self.vecenv_type == "dummy":
            envs = gym.vector.SyncVectorEnv(
                [
                    create_env(
                        self.env_cls,
                        problem_description,
                        env_specification,
                        pbs_per_env[i],
                    )
                    for i in range(self.num_envs)
                ],
            )
        elif self.vecenv_type == "subproc":
            print("self.env_cls", self.env_cls)
            envs = gym.vector.AsyncVectorEnv(
                [
                    create_env(
                        self.env_cls,
                        problem_description,
                        env_specification,
                        pbs_per_env[i],
                    )
                    for i in range(self.num_envs)
                ],
                # spwan helps when observation space is huge
                # context="spawn",
                copy=False,
            )
        elif self.vecenv_type == "graphgym":
            envs = AsyncGraphVectorEnv(
                [
                    create_env(
                        self.env_cls,
                        problem_description,
                        env_specification,
                        pbs_per_env[i],
                    )
                    for i in range(self.num_envs)
                ],
                # spwan helps when observation space is huge
                # and also with torch in subprocesses
                context="spawn",
                copy=False,
                shared_memory=True,
                disk=True,
            )

        print("... done creating environments")

        self.optimizer = self.optimizer_class(
            agent.parameters(), lr=lr, weight_decay=weight_decay
        )
        if opt_state_dict is not None:
            self.optimizer.load_state_dict(opt_state_dict)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        print("optimizer", self.optimizer)
        print("collecting rollouts using", rollout_agent_device)
        print("storing rollouts on", rollout_data_device)
        print("learning on", train_device)

        if not skip_model_trace:
            obs, info = self.validator.validation_envs[0].reset(soft=True)
            obs = agent.obs_as_tensor_add_batch_dim(obs)
            if agent.graphobs:
                torchinfo.summary(agent, depth=3, verbose=1)
            else:
                torchinfo.summary(agent, input_data=(obs,), depth=3, verbose=1)

        self.global_step = 0
        if not skip_initial_eval:
            print("initial validation")
            agent.to(rollout_agent_device)
            self.validator.validate(agent, self)
            print("... done initial validation")
        start_time = time.time()
        num_updates = self.total_timesteps // batch_size

        self.n_epochs = 0
        self.start_time = time.time()
        if self.return_based_scaling:
            sigma = None
        else:
            sigma = 1.0
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
                sigma,
            ) = self.collect_rollouts(
                agent, envs, env_specification, rollout_data_device, sigma
            )

            batch_size = b_logprobs.shape[0]
            # Optimizing the policy and value network
            b_inds = np.arange(batch_size)
            clipfracs = []

            entropy_losses = []
            pg_losses = []
            value_losses = []
            approx_kl_divs = []
            losses = []

            agent.to(train_device)
            for epoch in tqdm.tqdm(
                range(self.update_epochs), desc="   epochs             "
            ):
                self.n_epochs += 1
                np.random.shuffle(b_inds)
                self.optimizer.zero_grad()
                iter_it = 0

                approx_kl_divs_on_epoch = []
                for start in tqdm.tqdm(
                    range(0, batch_size, self.minibatch_size),
                    desc="   minibatches        ",
                    leave=False,
                ):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        agent.get_obs(b_obs, mb_inds),
                        action=b_actions.long()[mb_inds].to(train_device),
                        action_masks=b_action_masks[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds].to(train_device)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        approx_kl = ((ratio - 1) - logratio).mean()
                        approx_kl_divs.append(approx_kl.item())
                        if self.clip_coef is not None:
                            clipfracs += [
                                ((ratio - 1.0).abs() > self.clip_coef)
                                .float()
                                .mean()
                                .item()
                            ]
                    if self.target_kl is not None:
                        approx_kl_divs_on_epoch.append(approx_kl.item())

                    mb_advantages = agent.aggregate_reward(b_advantages[mb_inds]).to(
                        train_device
                    )
                    if self.norm_adv and mb_advantages.shape[0] > 1:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    if self.clip_coef is not None:
                        pg_loss2 = -mb_advantages * torch.clamp(
                            ratio,
                            1 - self.clip_coef,
                            1 + self.clip_coef,
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    else:
                        pg_loss = pg_loss1.mean()

                    # Value loss
                    if agent.agent_specification.two_hot is None:
                        if agent.agent_specification.symlog:
                            target = symlog(b_returns[mb_inds]).to(train_device)
                        else:
                            target = b_returns[mb_inds].to(train_device)
                        if self.critic_loss == "l2":
                            v_loss_unclipped = (
                                newvalue.view(-1, agent.reward_dim) - target
                            ) ** 2
                        elif self.critic_loss == "l1":
                            v_loss_unclipped = torch.abs(
                                newvalue.view(-1, agent.reward_dim) - target
                            )
                        elif self.critic_loss == "sl1":
                            v_loss_unclipped = torch.nn.functional.smooth_l1_loss(
                                newvalue.view(-1, agent.reward_dim),
                                target,
                                reduction="none",
                            )
                    else:
                        with torch.no_grad():
                            if agent.agent_specification.symlog:
                                twohot_target = calc_twohot(
                                    symlog(b_returns[mb_inds]).to(train_device), agent.B
                                )
                            else:
                                twohot_target = calc_twohot(
                                    b_returns[mb_inds].to(train_device), agent.B
                                )
                        v_loss_unclipped = nn.functional.cross_entropy(
                            newvalue, twohot_target, reduction="mean"
                        )

                    if self.clip_vloss:
                        v_clipped = b_values[mb_inds].to(train_device) + torch.clamp(
                            newvalue - b_values[mb_inds].to(train_device),
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        if self.critic_loss == "l2":
                            v_loss_clipped = (
                                v_clipped - b_returns[mb_inds].to(train_device)
                            ) ** 2
                        elif self.critic_loss == "l1":
                            v_loss_clipped = torch.abs(
                                v_clipped - b_returns[mb_inds].to(train_device)
                            )
                        elif self.critic_loss == "sl1":
                            v_loss_clipped = torch.nn.functional.smooth_l1_loss(
                                v_clipped,
                                b_returns[mb_inds].to(train_device),
                                reduction="none",
                            )
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        if self.critic_loss == "l2":
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = v_loss_max.mean()
                    else:
                        if agent.agent_specification.two_hot is not None:
                            v_loss = v_loss_unclipped
                        else:
                            if self.critic_loss == "l2":
                                v_loss = 0.5 * v_loss_unclipped.mean()
                            else:
                                v_loss = v_loss_unclipped.mean()
                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    )

                    losses.append(loss.item())
                    value_losses.append(v_loss.item())
                    pg_losses.append(pg_loss.item())
                    entropy_losses.append(entropy_loss.item())

                    loss.backward()
                    iter_it += 1
                    if iter_it == self.iter_size:
                        nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        iter_it = 0

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
                if self.clip_coef is not None:
                    self.logger.record("train/clip_fraction", np.mean(clipfracs))
                else:
                    self.logger.record("train/clip_fraction", 0.0)
                self.logger.record("train/loss", np.mean(losses))
                self.logger.record("train/explained_variance", explained_var)
                self.logger.record(
                    "train/n_epochs",
                    self.n_epochs,
                    exclude="tensorboard",
                )
                if self.clip_coef is not None:
                    self.logger.record("train/clip_range", self.clip_coef)
                else:
                    self.logger.record("train/clip_range", 0.0)
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

                ratio_to_ortools = np.array(self.validator.makespans) / np.array(
                    self.validator.ortools_makespans[
                        self.validator.default_ortools_strategy
                    ]
                )
                self.logger.record("train/ratio_monotony", monotony(ratio_to_ortools))
                self.logger.record("train/ratio_stability", stability(ratio_to_ortools))

            if (
                self.validation_freq is not None
                and iteration % self.validation_freq == 0
                and self.validator is not None
            ):
                self.validator.validate(agent, self)

                # Statistics from the agent validator.
                self.logger.record(
                    "validation/ppo_makespan",
                    self.validator.makespans[-1],
                )
                for ortools_strategy in self.validator.ortools_strategies:
                    self.logger.record(
                        f"validation/ortools_{ortools_strategy}_makespan",
                        self.validator.ortools_makespans[ortools_strategy][-1],
                    )
                self.logger.record(
                    "validation/random_makepsan",
                    self.validator.random_makespans[-1],
                )
                self.logger.record(
                    "validation/ratio_to_ortools",
                    self.validator.makespans[-1]
                    / self.validator.ortools_makespans[
                        self.validator.default_ortools_strategy
                    ][-1],
                )
                self.logger.record(
                    "validation/dist_to_ortools",
                    self.validator.makespans[-1]
                    - self.validator.ortools_makespans[
                        self.validator.default_ortools_strategy
                    ][-1],
                )
                for custom_agent in self.validator.custom_agents:
                    name = custom_agent.rule
                    self.logger.record(
                        f"validation/{name}",
                        self.validator.custom_makespans[name][-1],
                    )
                    self.logger.record(
                        f"validation/{name}_ratio_to_ortools",
                        self.validator.custom_makespans[name][-1]
                        / self.validator.ortools_makespans[
                            self.validator.default_ortools_strategy
                        ][-1],
                    )

            self.logger.dump(step=self.global_step)

        envs.close()

        ppo_makespans = np.array(self.validator.makespans)
        ortools_makespans = np.array(
            self.validator.ortools_makespans[self.validator.default_ortools_strategy]
        )
        ratios = ppo_makespans / ortools_makespans
        return np.min(ratios)
