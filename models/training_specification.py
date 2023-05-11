#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#    Antoine Jacquet <antoine.jacquet@jolibrain.com>
#    Michel Thomazo <thomazo.michel@gmail.com>
#    Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
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

import torch
from .dadapt_adam import DAdaptAdam
from .lion_pytorch import Lion


class TrainingSpecification:
    def __init__(
        self,
        total_timesteps,
        n_validation_env,
        fixed_validation,
        fixed_random_validation,
        validation_batch_size,
        validation_freq,
        display_env,
        path,
        custom_heuristic_name,
        ortools_strategy,
        max_time_ortools,
        scaling_constant_ortools,
        vecenv_type,
        validate_on_total_data,
        optimizer,
        n_workers,
        gamma,
        n_epochs,
        normalize_advantage,
        ent_coef,
        vf_coef,
        n_steps_episode,
        batch_size,
        iter_size,
        clip_range,
        target_kl,
        freeze_graph,
        lr,
        fe_lr,
        rpo,
        rpo_smoothing_param,
    ):
        self.lr = lr
        self.fe_lr = fe_lr
        self.total_timesteps = total_timesteps
        self.n_validation_env = n_validation_env
        self.fixed_validation = fixed_validation
        self.fixed_random_validation = fixed_random_validation
        self.validation_batch_size = validation_batch_size
        self.validation_freq = validation_freq
        self.display_env = display_env
        self.path = path
        self.custom_heuristic_name = custom_heuristic_name
        self.ortools_strategy = ortools_strategy
        self.max_time_ortools = max_time_ortools
        self.scaling_constant_ortools = scaling_constant_ortools
        self.vecenv_type = vecenv_type
        self.validate_on_total_data = validate_on_total_data
        self.optimizer = optimizer
        self.normalize_advantage = normalize_advantage
        self.n_steps_episode = n_steps_episode
        self.batch_size = batch_size
        self.iter_size = iter_size
        self.clip_range = clip_range
        self.target_kl = target_kl
        self.freeze_graph = freeze_graph
        self.rpo = rpo
        self.rpo_smoothing_param = rpo_smoothing_param

        if optimizer.lower() == "adam":
            self.optimizer_class = torch.optim.Adam
        elif optimizer.lower() == "sgd":
            self.optimizer_class = torch.optim.SGD
        elif optimizer.lower() == "adamw":
            self.optimizer_class = torch.optim.AdamW
        elif optimizer.lower() == "radam":
            self.optimizer_class = torch.optim.RAdam
        elif optimizer.lower() == "dadam":
            self.optimizer_class = DAdaptAdam
        elif optimizer.lower() == "lion":
            self.optimizer_class = Lion
        else:
            raise Exception("Optimizer not recognized")
        self.n_workers = n_workers
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

    def print_self(self):
        print(
            f"==========Training Description==========\n"
            f"Number of timesteps (total)       {self.total_timesteps}\n"
            f"Validation frequency:             {self.validation_freq}\n"
            f"Episodes per validation session:  {self.n_validation_env}\n"
            f"Validate on total data:           {self.validate_on_total_data}\n"
            f"Optimizer:                        {self.optimizer}\n"
            f"N workers:                        {self.n_workers}\n"
            f"Discount factor (gamma):          {self.gamma}\n"
            f"Number of epochs:                 {self.n_epochs}\n"
            f"Normalize advantage:              {self.normalize_advantage}\n"
            f"Entropy coefficient:              {self.ent_coef}\n"
            f"Value function coefficient:       {self.vf_coef}\n"
            f"Number steps per episode:         {self.n_steps_episode}\n"
            f"Batch size:                       {self.batch_size}\n"
            f"Iter size:                        {self.iter_size}\n"
            f"Clip Range:                       {self.clip_range}\n"
            f"Target KL:                        {self.target_kl}\n"
            f"Freeze graph:                     {self.freeze_graph}\n"
            f"Learning rate:                    {self.lr}\n"
            f"RPO:                              {self.rpo}\n"
            f"RPO smoothing:                    {self.rpo_smoothing_param}\n"
        )
