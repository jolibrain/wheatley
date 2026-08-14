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
import glob
import pickle

from generic.agent import calc_twohot, symexp, symlog
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import math
from alg.rollout_dataset import RolloutDataset, collate_rollout
from generic.utils import decode_mask, safe_mean, logcosh
from generic.adamw_schedulefree import AdamWScheduleFree
from generic.radam_schedulefree import RAdamScheduleFree
from generic.anon import Anon

from .logger import Logger, configure_logger, monotony, stability

import sys


class PPO:
    def __init__(
        self,
        training_specification,
        validator=None,
        discard_incomplete_trials=True,
        generate_duration_bounds=None,
        obstype=None,
    ):
        self.optimizer_class = training_specification.optimizer_class
        self.logger = configure_logger(
            folder=training_specification.path, format_strings=["json"]
        )
        self.generate_duration_bounds = generate_duration_bounds

        self.training_specification = training_specification
        self.obstype = obstype

        self.spo = training_specification.spo
        self.anon_gamma = training_specification.anon_gamma
        self.gamma = training_specification.gamma
        self.update_epochs = training_specification.n_epochs
        self.norm_adv = training_specification.normalize_advantage
        self.ent_coef = training_specification.ent_coef
        self.num_steps = training_specification.n_steps_episode
        self.gae_lambda = training_specification.gae_lambda
        self.clip_vloss = False
        self.clip_coef = training_specification.clip_range
        self.clip_coef_high = training_specification.clip_range_high
        self.vf_coef = training_specification.vf_coef
        self.target_kl = training_specification.target_kl
        self.minibatch_size = training_specification.batch_size
        self.iter_size = training_specification.iter_size
        self.validator = validator
        self.total_timesteps = training_specification.total_timesteps
        self.validation_freq = training_specification.validation_freq
        self.return_based_scaling = training_specification.return_based_scaling
        self.obs_on_disk = training_specification.store_rollouts_on_disk
        if training_specification.critic_loss == "l1":
            self.critic_loss = torch.nn.functional.l1_loss
        elif training_specification.critic_loss == "l2":
            self.critic_loss = torch.nn.functional.mse_loss
        elif training_specification.critic_loss == "sl1":
            self.critic_loss = torch.nn.functional.smooth_l1_loss
        elif training_specification.critic_loss == "logcosh":
            self.critic_loss = logcosh
        self.debug_net = training_specification.debug_net
        self.discard_incomplete_trials = discard_incomplete_trials
        self.max_shared_mem_per_worker = (
            training_specification.max_shared_mem_per_worker
        )

        self.max_grad_norm = training_specification.clip_grad_norm

        # in case of resume
        self._num_timesteps_at_start = 0
        self.ep_info_buffer = []
        self.espo = training_specification.espo

        if self.espo:
            self.clip_coef = None

    def collect_rollouts(
        self, agent, envs, num_envs, env_specification, data_device, sigma=1.0
    ):
        # ALGO Logic: Storage setup
        obs = []

        actions = torch.zeros(
            (
                self.num_steps,
                num_envs,
                agent.action_dim,
                agent.num_agents,
            ),
            # dtype=torch.long,
            device=data_device,
        )
        logprobs = torch.zeros(
            (self.num_steps, num_envs, agent.action_dim, agent.num_agents)
        ).to(data_device)
        rewards = torch.zeros((self.num_steps, num_envs, agent.reward_dim)).to(
            data_device
        )
        dones = torch.zeros((self.num_steps, num_envs)).to(data_device)
        values = torch.zeros((self.num_steps, num_envs, agent.reward_dim)).to(
            data_device
        )
        action_masks = list()

        if self.discard_incomplete_trials:
            to_keep = [[] for i in range(num_envs)]
            to_keep_candidate = [[] for i in range(num_envs)]

        # buffer filling
        o, info = envs.reset()
        # next obs is a list of dicts
        next_obs = agent.obs_as_tensor(o)
        action_mask = decode_mask(info["mask"])
        next_done = torch.zeros(num_envs).to(data_device)

        self.ep_info_buffer = []
        self.global_step += num_envs * self.num_steps

        if self.obs_on_disk is not None:
            for f in glob.glob(
                self.obs_on_disk + "/wheatley_" + str(os.getpid()) + "_*.obs"
            ):
                os.remove(f)

        for step in tqdm.tqdm(range(0, self.num_steps), desc="   collecting rollouts"):
            if self.obs_on_disk:
                for i, o in enumerate(next_obs):
                    fname = (
                        self.obs_on_disk
                        + "/wheatley_"
                        + str(os.getpid())
                        + "_"
                        + str(step * num_envs + i)
                        + ".obs"
                    )
                    o.save(fname)
                    obs.append(fname)
            else:
                obs.append(next_obs)
            current_mask = torch.as_tensor(action_mask, dtype=torch.bool)
            action_masks.append(current_mask)
            dones[step] = next_done

            if self.discard_incomplete_trials:
                for i in range(num_envs):
                    if dones[step][i] == 1:
                        to_keep[i].extend(to_keep_candidate[i])
                        to_keep_candidate[i].clear()
                    to_keep_candidate[i].append(step)

            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(
                    agent.preprocess(next_obs),
                    action_masks=current_mask,
                    deterministic=False,
                )
                value = agent.get_value_from_logits(value)

            values[step] = value.view(-1, agent.reward_dim).to(data_device)
            # print('logprob shape=', logprob.shape)
            # print('logprobs shape=', logprobs.shape)
            logprobs[step] = logprob  # .to(data_device)
            actions[step] = action  # .to(data_device)

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
            for i in range(num_envs):
                if next_done[i] == 1:
                    to_keep[i].extend(to_keep_candidate[i])

        with torch.no_grad():
            next_value = (
                agent.get_value_from_logits(agent.get_value(agent.preprocess(next_obs)))
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
                    [self.gamma] * (self.num_steps * num_envs - n_dones)
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
                dim=1,
            )
            for mask in action_masks
        ]
        action_masks = torch.stack(action_masks, dim=0)

        # flatten the batch
        b_obs = agent.rebatch_obs(obs)
        b_logprobs = logprobs.reshape(-1, agent.action_dim, agent.num_agents)
        b_actions = actions.reshape(
            -1,
            agent.action_dim,
            agent.num_agents,
        )
        b_advantages = advantages.reshape(-1, agent.reward_dim)

        b_returns = returns.reshape(-1, agent.reward_dim)
        b_values = values.reshape(-1, agent.reward_dim)
        b_action_masks = action_masks.reshape(-1, agent.num_agents, max_n_nodes)

        if self.discard_incomplete_trials:
            to_keep_b = [
                j + i * self.num_steps for i in range(num_envs) for j in to_keep[i]
            ]
            bobs_tokeep = list(b_obs[i] for i in to_keep_b)
            return RolloutDataset(
                agent,
                bobs_tokeep,
                b_logprobs[to_keep_b],
                b_actions[to_keep_b],
                b_advantages[to_keep_b],
                b_returns[to_keep_b],
                b_values[to_keep_b],
                b_action_masks[to_keep_b],
                sigma,
                self.obstype,
            )
        return RolloutDataset(
            agent,
            b_obs,
            b_logprobs,
            b_actions,
            b_advantages,
            b_returns,
            b_values,
            b_action_masks,
            sigma,
            self.obstype,
        )

    def train(
        self,
        agent,
        problem_description,
        env_specification,
        train_envs,
        num_envs,
        lr,
        weight_decay,
        log_interval=1,
        rollout_data_device=torch.device("cpu"),
        rollout_agent_device=torch.device("cpu"),
        train_device=torch.device("cpu"),
        opt_state_dict=None,
        skip_initial_eval=False,
        warmup=0,
        laber=None,
    ) -> float:
        # env setup
        batch_size = num_envs * self.num_steps
        # print("creating environments")

        print("... done creating environments")
        envs = train_envs

        if self.optimizer_class == torch.optim.RAdam:
            self.optimizer = self.optimizer_class(
                agent.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                decoupled_weight_decay=True,
            )
        elif self.optimizer_class == Anon:
            self.optimizer = self.optimizer_class(
                agent.parameters(),
                gamma=self.anon_gamma,
                lr=lr,
                weight_decay=weight_decay,
                weight_decouple=True,
            )
        else:
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

        self.global_step = 0
        if not skip_initial_eval:
            print("initial validation")
            agent.to(rollout_agent_device)
            agent.eval()
            self.validator.validate(agent, self, 0)
            print("... done initial validation")

        num_updates = self.total_timesteps // batch_size

        self.n_epochs = 0
        self.start_time = time.time()
        if self.return_based_scaling:
            sigma = None
        else:
            sigma = 1.0
        start = -warmup + 1 if warmup != 0 else 1
        for update in range(start, num_updates + 1):
            if update < 1:
                print("WARMUP", update + warmup)
            else:
                print("UPDATE ", update)
                if update == 1:
                    for g in self.optimizer.param_groups:
                        g["lr"] = lr

            agent.to(rollout_agent_device)
            agent.eval()
            if (
                self.optimizer_class == AdamWScheduleFree
                or self.optimizer_class == RAdamScheduleFree
            ):
                self.optimizer.eval()
            rollout_dataset = self.collect_rollouts(
                agent, envs, num_envs, env_specification, rollout_data_device, sigma
            )

            clipfracs = []
            entropy_losses = []
            pg_losses = []
            value_losses = []
            approx_kl_divs = []
            losses = []
            max_espo_dev = 0
            if self.debug_net:
                variances = {}
                grad_var = {}
                grad_mean = {}
                for n, p in agent.named_parameters():
                    if "bias" not in n:
                        variances[n] = []
                        if p.requires_grad:
                            grad_var[n] = []
                            grad_mean[n] = []

            agent.to(train_device)
            agent.train()
            if (
                self.optimizer_class == AdamWScheduleFree
                or self.optimizer_class == RAdamScheduleFree
            ):
                self.optimizer.train()
            for epoch in tqdm.tqdm(
                range(self.update_epochs), desc="   epochs             "
            ):
                self.n_epochs += 1
                self.optimizer.zero_grad()
                iter_it = 0

                clipped = 0
                unclipped = 0

                approx_kl_divs_on_epoch = []

                dataloader = torch.utils.data.DataLoader(
                    rollout_dataset,
                    batch_size=self.minibatch_size,
                    shuffle=True,
                    collate_fn=partial(collate_rollout, agent=agent),
                    num_workers=6,
                    pin_memory=True,
                )
                for (
                    batched_obs,
                    batched_logprobs,
                    batched_actions,
                    batched_advantages,
                    batched_returns,
                    batched_values,
                    batched_actions_masks,
                ) in tqdm.tqdm(
                    dataloader,
                    desc="   minibatches        ",
                    leave=False,
                ):
                    baction = batched_actions.to(train_device)
                    _, newlogprob, entropy, newvalue, unmasked_distrib = (
                        agent.get_action_and_value(
                            batched_obs,
                            action=baction,
                            action_masks=batched_actions_masks,
                        )
                    )
                    logratio = newlogprob - batched_logprobs.to(train_device)
                    ratio = logratio.exp()  # .squeeze(-1) #XXX(beniz): squeeze

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        approx_kl = (
                            (ratio - 1) - logratio
                        ).mean()  # TODO: maybe use the max instead
                        # approx_kl /= agent.action_dim
                        approx_kl_divs.append(approx_kl.item())
                        if self.clip_coef is not None:
                            clipfracs += [
                                # ((ratio - 1.0).abs() > self.clip_coef)
                                (
                                    torch.logical_or(
                                        (ratio - 1.0) < -self.clip_coef,
                                        (ratio - 1.0) > self.clip_coef_high,
                                    )
                                )
                                .float()
                                .mean()
                                .item()
                            ]
                    if self.target_kl is not None:
                        approx_kl_divs_on_epoch.append(approx_kl.item())

                    mb_advantages = agent.aggregate_reward(batched_advantages).to(
                        train_device
                    )
                    if self.norm_adv and mb_advantages.shape[0] > 1:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # add dim for number of agents and for action_dim
                    mba = -mb_advantages.unsqueeze(-1).unsqueeze(-1)
                    if self.spo:
                        spo_eps = 0.2  # as per  2401.16025
                        pg_loss = (
                            mba * ratio
                            + torch.abs(mba) * torch.pow(ratio - 1, 2) / (2 * spo_eps)
                        ).mean()
                    elif self.clip_coef is not None:
                        pg_loss1 = mba * ratio
                        pg_loss2 = mba * torch.clamp(
                            ratio,
                            1.0 - self.clip_coef,
                            1.0 + self.clip_coef_high,
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    else:
                        pg_loss1 = mba * ratio
                        espo_dev = (ratio - 1).abs().mean()
                        if max_espo_dev < espo_dev:
                            max_espo_dev = espo_dev
                        pg_loss = pg_loss1.mean()

                    # Value loss
                    # newvalue = newvalue.view(-1, 1)
                    if (
                        agent.agent_specification.two_hot is None
                        and agent.agent_specification.hl_gauss is None
                    ):
                        if agent.agent_specification.symlog:
                            target = symlog(batched_returns).to(train_device)
                        else:
                            target = batched_returns.to(train_device)
                        if laber is not None:
                            with torch.no_grad():
                                td_errors = (newvalue - target).abs().flatten()
                                priorities = torch.minimum(
                                    td_errors, torch.tensor([1]).to(td_errors.device)
                                )
                                indices = td_errors.multinomial(
                                    int(newvalue.shape[0] / laber), replacement=False
                                )
                                prios_for_indices = priorities[indices]
                                loss_weights = (
                                    1.0 / prios_for_indices
                                ) * priorities.mean()
                            newvalue = newvalue[indices]
                            target = target[indices]
                        else:
                            loss_weights = 1

                        v_loss_unclipped = (
                            loss_weights
                            * self.critic_loss(
                                newvalue,
                                target,
                                reduction="none",
                            )
                        ).mean()

                    elif agent.agent_specification.two_hot is not None:
                        with torch.no_grad():
                            if agent.agent_specification.symlog:
                                twohot_target = calc_twohot(
                                    symlog(batched_returns).to(train_device),
                                    agent.B,
                                )
                            else:
                                twohot_target = calc_twohot(
                                    batched_returns.to(train_device), agent.B
                                )
                        v_loss_unclipped = nn.functional.cross_entropy(
                            newvalue, twohot_target, reduction="mean"
                        )
                    else:  # hl_gaus case
                        with torch.no_grad():
                            hl_gauss_target = hl_gauss_to_probs(
                                batched_returns, agent.B
                            )
                        v_loss_unclipped = torch.nn.functional.cross_entropy(
                            newvalue, hl_gauss_target
                        )

                    if self.clip_vloss:
                        v_clipped = batched_values.to(train_device) + torch.clamp(
                            newvalue - batched_values.to(train_device),
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        if self.critic_loss == "l2":
                            v_loss_clipped = (
                                v_clipped - batched_returns.to(train_device)
                            ) ** 2
                        elif self.critic_loss == "l1":
                            v_loss_clipped = torch.abs(
                                v_clipped - batched_returns.to(train_device)
                            )
                        elif self.critic_loss == "sl1":
                            v_loss_clipped = torch.nn.functional.smooth_l1_loss(
                                v_clipped,
                                batched_returns.to(train_device),
                                reduction="none",
                            )
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = v_loss_max.mean()
                    else:
                        if agent.agent_specification.two_hot is not None:
                            v_loss = v_loss_unclipped
                        else:
                            v_loss = v_loss_unclipped
                    entropy_loss = entropy.mean()
                    # if update <= -1:
                    #     target_distrib = torch.ones_like(
                    #         unmasked_distrib.probs, device=unmasked_distrib.probs.device
                    #     )
                    #     target_distrib /= unmasked_distrib.probs.shape[1]
                    #     uniform_loss = torch.nn.functional.l1_loss(
                    #         unmasked_distrib.probs, target_distrib, reduction="sum"
                    #     )
                    loss = (
                        (pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef)
                        if update > 0
                        # else v_loss + 0.1 * uniform_loss
                        else v_loss
                    )

                    losses.append(loss.item())
                    value_losses.append(v_loss.item())
                    pg_losses.append(pg_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    loss.backward()

                    if self.debug_net:
                        for n, p in agent.named_parameters():
                            if "bias" not in n:
                                variances[n].append(p.var().item())
                                if p.requires_grad:
                                    grad_var[n].append(p.grad.var().item())
                                    grad_mean[n].append(p.grad.abs().mean().item())
                                else:
                                    print(n + " does not requires grad")

                    iter_it += 1
                    if iter_it == self.iter_size:
                        if self.max_grad_norm != 0.0:
                            unclipped_norm = nn.utils.clip_grad_norm_(
                                agent.parameters(), self.max_grad_norm
                            )
                            was_clipped_global = unclipped_norm > self.max_grad_norm
                            # global_clip_coef = min(
                            #     1.0, (self.max_grad_norm / unclipped_norm + 1e-6)
                            # )
                            # print(f"unclipped_norm : {unclipped_norm}")
                            # if was_clipped_global:
                            #     print(f"clip coef {global_clip_coef}")
                            # if not was_clipped_global or update < 3:
                            if not was_clipped_global:
                                unclipped += 1
                            else:
                                self.optimizer.step()
                                clipped += 1
                            # self.optimizer.step()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad()
                        iter_it = 0

                    # if self.clip_coef is None:
                    #     if espo_devs > 0.25:
                    #         break

                # print(f"unclipped {unclipped}  clipped {clipped}")

                if self.clip_coef is None:
                    if max_espo_dev > 0.25:
                        print(
                            f"\nstopping update due to espo devs too high after epoch {epoch} / {self.update_epochs}  (last espo_dev : {espo_dev})\n"
                        )
                        break

                elif self.target_kl is not None:
                    if np.mean(approx_kl_divs_on_epoch) > self.target_kl:
                        print(
                            f"stopping update due to too high kl divergence after epoch {epoch} / {self.update_epochs}  kl = {np.mean(approx_kl_divs_on_epoch)} > {self.target_kl}"
                        )
                        break

            y_pred, y_true = batched_values.cpu().numpy(), batched_returns.cpu().numpy()
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
                self.logger.record("train/return_variance", np.var(y_true))
                self.logger.record("train/value_variance", np.var(y_pred))
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
                        * num_envs
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

                if self.validator.compute_ortools:
                    ratio_to_ortools = np.array(self.validator.criterions) / np.array(
                        self.validator.ortools_criterions[
                            self.validator.default_ortools_strategy
                        ]
                    )
                    self.logger.record(
                        "train/ratio_monotony", monotony(ratio_to_ortools)
                    )
                    self.logger.record(
                        "train/ratio_stability", stability(ratio_to_ortools)
                    )
                if self.debug_net:
                    for k in variances.keys():
                        self.logger.record("net/var_" + k, np.mean(variances[k]))
                        self.logger.record("net/grad_var_" + k, np.mean(grad_var[k]))
                        self.logger.record("net/grad_mean_" + k, np.mean(grad_mean[k]))

            if (
                self.validation_freq is not None
                and iteration % self.validation_freq == 0
                and self.validator is not None
            ):
                self.validator.validate(agent, self, update)

                # Statistics from the agent validator.
                # self.logger.record(
                #     "validation/ppo_criterion",
                #     self.validator.criterions[-1],
                # )
                # if self.validator.compute_ortools:
                #     for ortools_strategy in self.validator.ortools_strategies:
                #         self.logger.record(
                #             f"validation/ortools_{ortools_strategy}_criterion",
                #             self.validator.ortools_criterions[ortools_strategy][-1],
                #         )
                # self.logger.record(
                #     "validation/random_makepsan",
                #     self.validator.random_criterions[-1],
                # )

                if self.validator.compute_ortools:
                    self.logger.record(
                        "validation/ratio_to_ortools",
                        self.validator.criterions[-1]
                        / self.validator.ortools_criterions[
                            self.validator.default_ortools_strategy
                        ][-1],
                    )
                    self.logger.record(
                        "validation/dist_to_ortools",
                        self.validator.criterions[-1]
                        - self.validator.ortools_criterions[
                            self.validator.default_ortools_strategy
                        ][-1],
                    )
                for custom_agent in self.validator.custom_agents:
                    name = custom_agent.rule
                    self.logger.record(
                        f"validation/{name}",
                        self.validator.custom_criterions[name][-1],
                    )
                    if self.validator.compute_ortools:
                        self.logger.record(
                            f"validation/{name}_ratio_to_ortools",
                            self.validator.custom_criterions[name][-1]
                            / self.validator.ortools_criterions[
                                self.validator.default_ortools_strategy
                            ][-1],
                        )

            self.logger.dump(step=self.global_step)

        envs.close()

        # ppo_criterions = np.array(self.validator.criterions)
        # if self.validator.compute_ortools:
        #     ortools_criterions = np.array(
        #         self.validator.ortools_criterions[
        #             self.validator.default_ortools_strategy
        #         ]
        #     )
        #     ratios = ppo_criterions / ortools_criterions
        # else:
        #     ratios = ppo_criterions / ppo_criterions[0]
        # return np.min(ratios)
