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


from functools import partial

import numpy as np
import torch
from torch.distributions.categorical import Categorical


class Agent(torch.nn.Module):
    def __init__(
        self,
        env_specification,
        gnn=None,
        value_net=None,
        action_net=None,
        agent_specification=None,
    ):
        """
        There are 2 ways to init an Agent:
         - Either provide a valid env_specification and agent_specification
         - Or use the load method, to load an already saved Agent
        """
        super().__init__()
        # User must provide an agent_specification or a model at least.
        if agent_specification is None or env_specification is None:
            raise Exception(
                "Please provide an agent_specification to create a new Agent"
            )

        self.env_specification = env_specification
        self.agent_specification = agent_specification

    @staticmethod
    def init_weights(module, gain=1, zero_bias=True, ortho_embed=False) -> None:
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None and zero_bias:
                module.bias.data.fill_(0.0)
        if ortho_embed and isinstance(module, torch.nn.Embedding):
            torch.nn.init.orthogonal_(module.weight, gain=gain)

    def save(self, path):
        """Saving an agent corresponds to saving his model and a few args to specify how the model is working"""
        device = next(self.gnn.parameters()).device
        self.to(torch.device("cpu"))
        torch.save(
            {
                "env_specification": self.env_specification,
                "agent_specification": self.agent_specification,
                "gnn": self.gnn.state_dict(),
                "value_net": self.value_net.state_dict(),
                "action_net": self.action_net.state_dict(),
            },
            path,
        )
        self.to(device)

    @classmethod
    def load(cls, path):
        pass

    # def critic(self, x):
    #     features = self.gnn(x)
    #     # filter out node specific features
    #     return self.value_net(features[:, 0, features.shape[2] // 2 :])

    # def actor(self, x):
    #     return self.action_net(self.gnn(x))

    def get_value(self, x):
        features = self.gnn(x)
        # filter out node specific features
        return self.value_net(features[:, 0, features.shape[2] // 2 :])

    def get_action_and_value(
        self, x, action=None, action_masks=None, deterministic=False
    ):
        features = self.gnn(x)
        value = self.value_net(features[:, 0, features.shape[2] // 2 :])
        logits = self.action_net(features).squeeze(-1)
        if action_masks is not None:
            mask = torch.as_tensor(
                action_masks, dtype=torch.bool, device=features.device
            )
            HUGE_NEG = torch.tensor(-1e12, dtype=logits.dtype, device=features.device)
            logits = torch.where(mask, logits, HUGE_NEG)
        distrib = Categorical(logits=logits)
        if action is None:
            if deterministic == False:
                action = distrib.sample()
            else:
                action = torch.argmax(distrib.probs, dim=1)
        if action_masks is not None:
            p_log_p = distrib.logits * distrib.probs
            p_log_p = torch.where(mask, p_log_p, torch.tensor(0.0).to(features.device))
            entropy = -p_log_p.sum(-1)
        else:
            entropy = distrib.entropy()
        return action, distrib.log_prob(action), entropy, value

    def get_action_probs_and_value(self, x, action_masks):
        features = self.gnn(x)
        value = self.value_net(features[:, 0, features.shape[2] // 2 :])
        action_logits = self.action_net(features).squeeze(-1)
        if action_masks is not None:
            mask = torch.as_tensor(
                action_masks, dtype=torch.bool, device=features.device
            )
            HUGE_NEG = torch.tensor(
                -1e12, dtype=action_logits.dtype, device=features.device
            )
            logits = torch.where(mask, action_logits, HUGE_NEG)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            return probs, value
        # distrib = Categorical(logits=action_logits)
        # return distrib.probs
        probs = torch.nn.functional.softmax(action_logits, dim=-1)
        return probs, value

    def predict(self, observation, deterministic, action_masks):
        with torch.no_grad():
            features = self.gnn(observation)
            logits = self.action_net(features)
            if action_masks is not None:
                mask = torch.as_tensor(
                    action_masks, dtype=torch.bool, device=features.device
                ).reshape(logits.shape)
                HUGE_NEG = torch.tensor(
                    -1e12, dtype=logits.dtype, device=features.device
                )
                logits = torch.where(mask, logits, HUGE_NEG)
            distrib = Categorical(logits=logits.squeeze(-1))
            if deterministic == False:
                action = distrib.sample()
            else:
                action = torch.argmax(distrib.probs, dim=1)
            return action

    def solve(self, problem_description):
        # Creating an environment on which we will run the inference
        env = Env(problem_description, self.env_specification)

        # Running the inference loop
        observation, info = env.reset()
        action_masks = info["mask"]
        done = False
        while not done:
            action_masks = get_action_masks(env)
            action = self.predict(
                observation, deterministic=True, action_masks=action_masks
            )
            observation, reward, done, _, info = env.step(action)
            mask = info["mask"]

        return env.get_solution()

    def forward(self, observation, action=None, action_masks=None, deterministic=False):
        return self.get_action_and_value(
            observation, action, action_masks, deterministic
        )
