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
    ):
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

    def print_self(self):
        print(
            f"==========Training Description==========\n"
            f"Number of timesteps (total)       {self.total_timesteps}\n"
            f"Validation frequency:             {self.validation_freq}\n"
            f"Episodes per validation session:  {self.n_validation_env}\n"
            f"Validate on total data:           {self.validate_on_total_data}\n"
        )
