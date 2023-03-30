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

from utils.utils import get_n_features


class JSSPEnvSpecification:
    def __init__(
        self,
        max_n_jobs,
        max_n_machines,
        normalize_input,
        input_list,
        insertion_mode,
        max_edges_factor,
        sample_n_jobs,
        chunk_n_jobs,
        observe_conflicts_as_cliques,
        observe_real_duration_when_affect,
        do_not_observe_updated_bounds,
    ):
        self.max_n_jobs = max_n_jobs
        self.max_n_machines = max_n_machines
        self.max_n_nodes = max_n_jobs * max_n_machines
        self.max_n_edges = self.max_n_nodes**2
        self.normalize_input = normalize_input
        self.input_list = input_list
        self.insertion_mode = insertion_mode
        self.max_edges_factor = max_edges_factor
        self.sample_n_jobs = sample_n_jobs
        self.chunk_n_jobs = chunk_n_jobs
        self.add_boolean = (insertion_mode == "choose_forced_insertion") or (
            insertion_mode == "slot_locking"
        )
        self.n_features = get_n_features(
            self.input_list, self.max_n_jobs, self.max_n_machines
        )
        self.observe_conflicts_as_cliques = observe_conflicts_as_cliques
        self.observe_real_duration_when_affect = observe_real_duration_when_affect
        self.do_not_observe_updated_bounds = do_not_observe_updated_bounds

    def print_self(self):
        print_input_list = [
            el.lower().title().replace("_", " ") for el in self.input_list
        ]
        print(
            f"==========Env Description     ==========\n"
            f"Max size:                           {self.max_n_jobs} x {self.max_n_machines}\n"
            f"Input normalization:                {'Yes' if self.normalize_input else 'No'}\n"
            f"Insertion mode:                     {self.insertion_mode.lower().title().replace('_', ' ')}\n"
            f"Observe real duration when affect:  {self.observe_real_duration_when_affect}\n"
            f"Do not observe tct:                 {self.do_not_observe_updated_bounds}\n"
            f"List of features:\n - Task Completion Times\n - Machine Id"
        )
        print(" - " + "\n - ".join(print_input_list) + "\n")
