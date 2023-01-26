from functools import partial
import numpy as np
from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy
from torch import nn
import torch
from torch.distributions.utils import probs_to_logits, logits_to_probs
from .mlp import MLP


class GraphExtractor(nn.Module):
    def __init__(self):
        super(GraphExtractor, self).__init__()

    def forward(self, features):
        return features, features[:, 0, features.shape[2] // 2 :]

    def forward_actor(self, features):
        return features

    def forward_critic(self, features):
        # filter out node specific features
        return features[:, 0, features.shape[2] // 2 :]


class Policy(MaskableActorCriticPolicy):
    def extract_features(self, obs):
        # skip preprocessing
        return self.features_extractor(obs)

    def _build(self, lr_schedule):
        """
        Create the networks and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.mlp_extractor = GraphExtractor()

        self.action_net = MLP(
            len(self.net_arch[0]["pi"]),
            self.features_dim,
            self.net_arch[0]["pi"][0],
            1,
            False,
            self.activation_fn,
            self.device,
        )
        self.value_net = MLP(
            len(self.net_arch[0]["vf"]),
            self.features_dim // 2,
            self.net_arch[0]["vf"][0],
            1,
            False,
            self.activation_fn,
            self.device,
        )

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                # self.features_extractor: np.sqrt(2),
                self.features_extractor: 1,
                self.action_net: 1,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        fe_lr = self.optimizer_kwargs["fe_lr"] if self.optimizer_kwargs["fe_lr"] is not None else self.optimizer_kwargs["lr"]
        pgroup = [
            {"params": self.features_extractor.parameters(), "lr": fe_lr},
        ]

        self.optimizer = self.optimizer_class(pgroup, lr=lr_schedule(1))
        print("optimizer", self.optimizer)


class RPOPolicy(Policy):
    def set_rpo_smoothing_param(self, sp):
        self.rpo_smoothing_param = sp

    def _get_action_dist_from_latent(self, latent_pi):
        action_logits = self.action_net(latent_pi)
        action_probs = logits_to_probs(action_logits.view(-1, action_logits.shape[-2]))
        ru = torch.rand((action_logits.shape[0]), device=action_logits.device)
        try:
            ru *= self.rpo_smoothing_param
        except AttributeError:
            # use 1 as a default
            pass
        ru = ru.unsqueeze(-1)
        action_probs = (action_probs + ru) / (1 + ru * action_logits.shape[-2])
        new_logits = probs_to_logits(action_probs).unsqueeze(-1)
        return self.action_dist.proba_distribution(action_logits=new_logits)
