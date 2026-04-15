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

from gymnasium.spaces import Discrete, Dict, Box, Sequence, Text
import numpy as np


class EnvSpecification:
    def __init__(
        self,
        problems,
        normalize_input,
        sample_n_jobs,
        chunk_n_jobs,
        add_rp_edges,
        observe_real_duration_when_affect,
        do_not_observe_updated_bounds,
        factored_rp,
        remove_old_resource_info,
        remove_past_prec,
        observation_horizon_step,
        observation_horizon_time,
        fast_forward,
        observe_subgraph,
        random_taillard,
        max_n_modes=None,
    ):
        self.problems = problems
        if max_n_modes is None:
            self.max_n_modes = self.problems.max_n_modes
        else:
            self.max_n_modes = max_n_modes
        self.max_n_nodes = self.max_n_modes
        self.max_n_jobs = self.problems.max_n_jobs
        self.max_n_resources = self.problems.max_n_resources
        self.max_resource_request = self.problems.max_resource_request
        self.max_resource_availability = self.problems.max_resource_availability
        self.normalize_input = normalize_input
        self.factored_rp = factored_rp
        self.add_rp_edges = add_rp_edges
        self.sample_n_jobs = sample_n_jobs
        self.chunk_n_jobs = chunk_n_jobs
        self.add_boolean = False
        self.observe_real_duration_when_affect = observe_real_duration_when_affect
        self.do_not_observe_updated_bounds = do_not_observe_updated_bounds
        self.remove_old_resource_info = remove_old_resource_info
        self.remove_past_prec = remove_past_prec
        self.observation_horizon_step = observation_horizon_step
        self.observation_horizon_time = observation_horizon_time
        self.fast_forward = fast_forward
        self.observe_subgraph = observe_subgraph
        self.random_taillard = random_taillard

    def print_self(self):
        print(
            f"==========Env Description     ==========\n"
            f"Max size:                           {self.max_n_modes}\n"
            f"Max n ressources:                   {self.max_n_resources}\n"
            f"Input normalization:                {'Yes' if self.normalize_input else 'No'}\n"
            f"Observe real duration when affect:  {self.observe_real_duration_when_affect}\n"
            f"Do not observe tct:                 {self.do_not_observe_updated_bounds}\n"
            f"add resource prcedence edges:       {self.add_rp_edges}\n"
            f"remove old resource info:           {self.remove_old_resource_info}\n"
            f"remove past prec:                   {self.remove_past_prec}\n"
            f"observation horizon step:           {self.observation_horizon_step}\n"
            f"observation horizon time:           {self.observation_horizon_time}\n"
            f"fast forward:                       {self.fast_forward}\n"
            f"observe subgraph:                   {self.observe_subgraph}\n"
        )
