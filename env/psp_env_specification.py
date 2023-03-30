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


class PSPEnvSpecification:
    def __init__(
        self,
        problems,
        normalize_input,
        input_list,
        max_edges_factor,
        sample_n_jobs,
        chunk_n_jobs,
        observe_conflicts_as_cliques,
        observe_real_duration_when_affect,
        do_not_observe_updated_bounds,
    ):
        self.problems = problems
        self.max_n_modes = self.problems.max_n_modes
        self.max_n_nodes = self.max_n_modes
        self.max_n_edges = self.max_n_nodes**2
        self.max_n_resources = self.problems.max_n_resources
        self.max_resource_request = self.problems.max_resource_request
        self.max_resource_availability = self.problems.max_resource_availability
        self.normalize_input = normalize_input
        self.input_list = input_list
        if "selectable" in input_list:
            self.input_list.remove("selectable")
        self.max_edges_factor = max_edges_factor
        self.sample_n_jobs = sample_n_jobs
        self.chunk_n_jobs = chunk_n_jobs
        self.add_boolean = False
        self.n_features = self.get_n_features()
        self.observe_conflicts_as_cliques = observe_conflicts_as_cliques
        self.observe_real_duration_when_affect = observe_real_duration_when_affect
        self.do_not_observe_updated_bounds = do_not_observe_updated_bounds

    def get_n_features(self):
        # 4 for task completion times, 1 for is_affected
        # 1 for mandatory selectable
        n_features = 6
        # level of resource used by every node
        n_features += self.max_n_resources
        # most features make 4 values
        n_features += 4 * len(self.input_list)
        return n_features

    def print_self(self):
        print_input_list = [
            el.lower().title().replace("_", " ") for el in self.input_list
        ]
        print(
            f"==========Env Description     ==========\n"
            f"Max size:                           {self.max_n_modes}\n"
            f"Input normalization:                {'Yes' if self.normalize_input else 'No'}\n"
            f"Observe real duration when affect:  {self.observe_real_duration_when_affect}\n"
            f"Do not observe tct:                 {self.do_not_observe_updated_bounds}\n"
            f"List of features:\n - Task Completion Times - selectable"
        )
        print(" - " + "\n - ".join(print_input_list) + "\n")
