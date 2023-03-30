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

from problem.jssp_description import JSSPDescription as ProblemDescription
from env.jssp_env_specification import JSSPEnvSpecification
from models.agent_specification import AgentSpecification
from models.jssp_agent import JSSPAgent as Agent
from utils.utils import rebatch_obs, get_obs
from utils.ortools import *
from env.env import Env
import numpy as np
import torch
from torch.distributions.categorical import Categorical


class Pretrainer:
    def __init__(
        self,
        problem_description,
        env_specification,
        agent_specification,
        num_envs=1,
        prob=0.9,
    ):
        self.problem_description = problem_description
        self.env_specification = env_specification
        self.agent_specification = agent_specification
        self.num_envs = num_envs
        self.prob = prob

    def get_target_probs(self, all_masks, all_past_actions, pa_to_a, mb_inds):
        masks = all_masks[mb_inds]

        # discard precise stats from pa_to_a because it comes from np.random
        # , uniform on present actions, sum = prob
        # other non masks get rest

        p = torch.where(masks, 1, 0)
        l = torch.empty(masks.shape)
        for n, i in enumerate(mb_inds):
            sum_non_masked = p[i].sum()
            pa = all_past_actions[i]
            actions = pa_to_a[tuple(pa)]
            nactions = len(actions)
            if sum_non_masked == 1:
                rest = 0
                real_prob = 1
            else:
                rest = (1 - self.prob) / (sum_non_masked - nactions)
                real_prob = self.prob / nactions
            r = torch.where(p[i] == 1, rest, 0)
            for a in actions:
                r[a] = real_prob
            l[n] = r
        return l

    def pretrain(self, agent, num_epochs, minibatch_size, n_traj, lr=0.0002):

        all_obs = []
        all_masks = []
        all_actions = []
        all_past_actions = []
        for e in range(self.num_envs):
            env = Env(self.problem_description, self.env_specification)
            for i in range(n_traj):
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

        b_inds = np.arange(len(actions))

        optimizer = self.agent_specification.optimizer_class(agent.parameters(), lr=lr)
        for epoch in range(num_epochs):
            # np.random.shuffle(b_inds)
            eloss = 0
            for start in range(0, len(actions), minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                action_probs = agent.get_action_probs(
                    get_obs(all_obs, mb_inds), all_masks[mb_inds]
                )
                target_probs = self.get_target_probs(
                    all_masks, all_past_actions, pa_to_a, mb_inds
                ).to(action_probs.device)
                loss = torch.nn.functional.mse_loss(
                    action_probs, target_probs, reduction="sum"
                )
                eloss += loss.item()
                loss.backward()
                optimizer.step()
            print("pretrain loss:", eloss / len(actions))
