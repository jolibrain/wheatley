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

import numpy as np

from utils.utils import (
    generate_deterministic_problem,
    generate_problem_distrib,
    generate_problem_durations,
)


class PSPDescription:
    def __init__(
        self,
        transition_model_config,
        reward_model_config,
        deterministic,
        train_psps,
        test_psps,
    ):
        """ """

        self.transition_model_config = transition_model_config
        self.reward_model_config = reward_model_config
        self.deterministic = deterministic
        self.train_psps = train_psps
        self.test_psps = test_psps

        self.max_n_jobs = max([psp["n_jobs"] for psp in train_psps + test_psps])
        self.max_n_modes = max([psp["n_modes"] for psp in train_psps + test_psps])
        self.max_n_resources = max(
            [psp["n_resources"] for psp in train_psps + test_psps]
        )

    def print_self(self):
        print(
            f"==========Problem Description ==========\n"
            f"Number of jobs:                   {self.max_n_jobs}\n"
            f"number of modes:                  {self.max_n_modes}\n"
            f"Number of resources:              {self.max_n_resources}\n"
            f"Transition model:                 {self.transition_model_config}\n"
            f"Reward model:                     {self.reward_model_config}\n"
            f"Deterministic/Stochastic:         {'Deterministic' if self.deterministic else 'Stochastic'}\n"
        )
