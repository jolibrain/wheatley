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

import numpy as np
import torch
import visdom
from torch.distributions.categorical import Categorical
from tqdm import tqdm

from env.jssp_env_specification import JSSPEnvSpecification
from models.jssp_agent import JSSPAgent as Agent
from models.training_specification import TrainingSpecification
from problem.jssp_description import JSSPDescription as ProblemDescription
from utils.ortools import *
from utils.utils import get_obs, rebatch_obs


class Pretrainer:
    def __init__(
        self,
        problem_description,
        env_specification,
        training_specification,
        env_cls,
        num_envs=1,
        num_eval_envs=1,
        trajectories=10,
        prob=0.9,
    ):
        self.problem_description = problem_description
        self.env_specification = env_specification
        self.training_specification = training_specification
        self.env_cls = env_cls
        self.num_envs = num_envs
        self.num_eval_envs = num_eval_envs
        self.trajectories = trajectories
        self.prob = prob

        self.vis = visdom.Visdom(env=self.training_specification.display_env)
        self.train_losses = []
        self.eval_losses = []

    def get_target_probs(self, all_masks, all_past_actions, pa_to_a, mb_inds):
        masks = all_masks[mb_inds]

        # discard precise stats from pa_to_a because it comes from np.random
        # , uniform on present actions, sum = prob
        # other non masks get rest

        p = torch.where(masks, 1, 0)
        l = torch.empty(masks.shape)
        for n, i in enumerate(mb_inds):
            sum_non_masked = p[n].sum()
            pa = all_past_actions[i]
            actions = pa_to_a[tuple(pa)]
            nactions = len(actions)
            if sum_non_masked == 1:
                rest = 0
                real_prob = 1
            else:
                rest = (1 - self.prob) / (sum_non_masked - nactions)
                real_prob = self.prob / nactions
            r = torch.where(p[n] == 1, rest, 0)
            for a in actions:
                r[a] = real_prob
            l[n] = r
        return l

    def generate_dataset_(self, num_envs: int, trajectories: int) -> tuple:
        all_obs = []
        all_masks = []
        all_actions = []
        all_past_actions = []
        env = self.env_cls(self.problem_description, self.env_specification, 0)

        for e in tqdm(range(num_envs), desc="Generating pretraining dataset"):
            env.reset()
            for _ in tqdm(range(trajectories), desc="Trajectories", leave=False):
                if self.problem_description.deterministic:
                    env.reset(soft=True)
                else:
                    # WARNING: This works only for fixed problems. There should be
                    # a way to do this with non-fixed problems.
                    env.reset()  # Sample new durations within the bounds.
                (
                    obs,
                    masks,
                    actions,
                    past_actions,
                ) = get_ortools_trajectory_and_past_actions(env)
                all_obs.extend(obs)
                all_masks.extend(masks)
                all_actions.extend(actions)
                all_past_actions.extend(past_actions)

        all_obs = rebatch_obs(all_obs)
        all_masks = torch.tensor(np.array(all_masks))
        all_actions = torch.tensor(all_actions)

        pa_to_a = {}
        for pa, a in zip(all_past_actions, all_actions):
            key = tuple(pa)
            if key in pa_to_a:
                pa_to_a[key].add(a.long().item())
            else:
                pa_to_a[key] = set([a.long().item()])

        return all_obs, all_masks, all_actions, all_past_actions, pa_to_a

    def pretrain(self, agent, num_epochs, minibatch_size, lr=0.0002):
        (
            train_obs,
            train_masks,
            train_actions,
            train_past_actions,
            train_pa_to_a,
        ) = self.generate_dataset_(self.num_envs, self.trajectories)

        (
            eval_obs,
            eval_masks,
            eval_actions,
            eval_past_actions,
            eval_pa_to_a,
        ) = self.generate_dataset_(self.num_eval_envs, max(self.trajectories // 10, 1))

        b_inds = np.arange(len(train_actions))

        optimizer = self.training_specification.optimizer_class(
            agent.parameters(), lr=lr
        )
        for _ in tqdm(range(num_epochs), desc="Pretrain"):
            np.random.shuffle(b_inds)

            eloss = 0
            for start in tqdm(
                range(0, len(train_actions), minibatch_size),
                desc="Batches",
                leave=False,
            ):
                end = min(start + minibatch_size, len(train_actions))
                mb_inds = b_inds[start:end]

                action_probs = agent.get_action_probs(
                    get_obs(train_obs, mb_inds), train_masks[mb_inds]
                )
                target_probs = self.get_target_probs(
                    train_masks, train_past_actions, train_pa_to_a, mb_inds
                ).to(action_probs.device)
                loss = torch.nn.functional.mse_loss(
                    action_probs,
                    target_probs,
                    reduction="none",
                )
                # loss = torch.nn.functional.kl_div(
                #     action_probs,
                #     target_probs,
                #     log_target=False,
                #     reduction="none",
                # )
                eloss += loss.sum().item()
                loss.mean().backward()
                optimizer.step()

            self.train_losses.append(eloss / len(train_actions))

            eloss = 0
            with torch.inference_mode():
                for start in range(0, len(eval_actions), minibatch_size):
                    end = min(start + minibatch_size, len(eval_actions))
                    mb_inds = np.arange(start, end)

                    action_probs = agent.get_action_probs(
                        get_obs(eval_obs, mb_inds), eval_masks[mb_inds]
                    )
                    target_probs = self.get_target_probs(
                        eval_masks, eval_past_actions, eval_pa_to_a, mb_inds
                    ).to(action_probs.device)
                    loss = torch.nn.functional.mse_loss(
                        action_probs, target_probs, reduction="sum"
                    )
                    # loss = torch.nn.functional.kl_div(
                    #     action_probs,
                    #     target_probs,
                    #     log_target=False,
                    #     reduction="sum",
                    # )
                    eloss += loss.item()

            if self.eval_losses == [] or (
                min(self.eval_losses) > eloss / len(eval_actions)
            ):
                pretrain_model_path = self.training_specification.path + "pretrain.pkl"
                agent.save(pretrain_model_path)

            self.eval_losses.append(eloss / len(eval_actions))

            Y = np.array([self.train_losses, self.eval_losses])
            self.vis.line(
                Y=Y.T,
                X=np.arange(len(self.train_losses)),
                win="pretrain-loss",
                opts={"title": "Pretrain Loss", "legend": ["train", "eval"]},
            )
