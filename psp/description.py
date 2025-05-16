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


class Description:
    def __init__(
        self,
        transition_model_config,
        reward_model_config,
        deterministic,
        generate_duration_bounds,
        train_psps,
        test_psps,
        seed,
        unload=True,
    ):
        """ """

        self.transition_model_config = transition_model_config
        self.reward_model_config = reward_model_config
        self.deterministic = deterministic
        self.generate_duration_bounds = generate_duration_bounds
        self.train_psps = train_psps
        self.test_psps = test_psps
        self.seed = seed
        self.unload = unload

        if isinstance(train_psps[0], dict):
            self.max_n_jobs = max([psp["n_jobs"] for psp in train_psps + test_psps])
            self.max_n_modes = max([psp["n_modes"] for psp in train_psps + test_psps])

            self.max_n_resources = max(
                [psp["n_resources"] for psp in train_psps + test_psps]
            )
            self.max_resource_request = max(
                [psp["max_resource_request"] for psp in train_psps + test_psps]
            )
            self.max_resource_availability = max(
                [psp["max_resource_availability"] for psp in train_psps + test_psps]
            )
        else:
            self.max_n_jobs = max(
                [len(psp.job_labels) for psp in train_psps + test_psps]
            )
            self.max_n_modes = max([psp.n_modes for psp in train_psps + test_psps])

            self.max_n_resources = max(
                [psp.n_resources for psp in train_psps + test_psps]
            )
            self.max_resource_request = max(
                [psp.max_resource_consumption for psp in train_psps + test_psps]
            )
            self.max_resource_availability = max(
                [psp.max_resource_availability for psp in train_psps + test_psps]
            )
            if self.unload:
                self.train_psps_ids = [psp.pb_id for psp in train_psps]
                self.train_psps = None
                self.test_psps_ids = [psp.pb_id for psp in test_psps]
                self.test_psps = None
                self.ntrain = len(self.train_psps_ids)
                self.ntest = len(self.test_psps_ids)
            else:
                self.ntrain = len(self.train_psps)
                self.ntest = len(self.test_psps)

    def print_self(self):
        print(
            f"==========Problem Description ==========\n"
            f"Number of jobs:                   {self.max_n_jobs}\n"
            f"number of modes:                  {self.max_n_modes}\n"
            f"Number of resources:              {self.max_n_resources}\n"
            f"Transition model:                 {self.transition_model_config}\n"
            f"Reward model:                     {self.reward_model_config}\n"
            f"Deterministic/Stochastic:         {'Deterministic' if self.deterministic else 'Stochastic'}\n"
            f"Generate Duration Bounds:         {self.generate_duration_bounds}\n"
            f"Train set size:                   {self.ntrain}\n"
            f"Test set size:                    {self.ntest}\n"
        )
