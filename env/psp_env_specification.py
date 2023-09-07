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

from gymnasium.spaces import Discrete, Dict, Box
import numpy as np


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
        add_rp_edges,
        observe_real_duration_when_affect,
        do_not_observe_updated_bounds,
        factored_rp,
        remove_old_resource_info,
        remove_past_prec,
    ):
        self.problems = problems
        self.max_n_modes = self.problems.max_n_modes
        self.max_n_nodes = self.max_n_modes
        self.max_n_jobs = self.problems.max_n_jobs
        self.max_n_edges = self.max_n_nodes**2
        self.max_n_resources = self.problems.max_n_resources
        self.max_resource_request = self.problems.max_resource_request
        self.max_resource_availability = self.problems.max_resource_availability
        self.normalize_input = normalize_input
        self.input_list = input_list
        self.factored_rp = factored_rp
        self.add_rp_edges = add_rp_edges
        if "selectable" in input_list:
            self.input_list.remove("selectable")
        if "duration" in input_list:
            self.input_list.remove("duration")
        self.max_edges_factor = max_edges_factor
        self.sample_n_jobs = sample_n_jobs
        self.chunk_n_jobs = chunk_n_jobs
        self.add_boolean = False
        self.n_features = self.get_n_features()
        self.observe_conflicts_as_cliques = observe_conflicts_as_cliques
        self.observe_real_duration_when_affect = observe_real_duration_when_affect
        self.do_not_observe_updated_bounds = do_not_observe_updated_bounds
        self.action_space = Discrete(self.max_n_nodes)
        self.remove_old_resource_info = remove_old_resource_info
        self.remove_past_prec = remove_past_prec

        if self.max_edges_factor > 0:
            self.shape_pr = (
                2,
                self.max_edges_factor * self.max_n_nodes,
            )
            self.shape_rc = (
                2,
                self.max_edges_factor * self.max_n_nodes * self.max_n_resources * 2,
            )
            self.shape_rc_att = (
                self.max_edges_factor * self.max_n_nodes * self.max_n_resources * 2,
                3,
            )
            if self.add_rp_edges != "none":
                self.shape_rp = (
                    2,
                    self.max_edges_factor * self.max_n_nodes * self.max_n_resources * 2,
                )
                if self.factored_rp:
                    self.shape_rp_att = (
                        self.max_edges_factor
                        * self.max_n_nodes
                        * self.max_n_resources
                        * 2,
                        self.max_n_resources * 3,
                    )
                else:
                    self.shape_rp_att = (
                        self.max_edges_factor
                        * self.max_n_nodes
                        * self.max_n_resources
                        * 2,
                        4,
                    )

        else:
            self.shape_pr = (2, self.max_n_edges)
            self.shape_rc = (
                2,
                self.max_n_edges * self.max_n_resources,
            )
            self.shape_rc_att = (
                # rid, rval
                self.max_n_edges * self.max_n_resources,
                3,
            )
            if self.add_rp_edges != "none":
                self.shape_rp = (
                    2,
                    self.max_n_edges * self.max_n_resources,
                )
                if self.factored_rp:
                    self.shape_rp_att = (
                        self.max_n_edges * self.max_n_resources,
                        self.max_n_resources * 3,
                    )
                else:
                    self.shape_rp_att = (
                        self.max_n_edges * self.max_n_resources,
                        4,
                    )

        # self.observation_space = Dict(
        dict_specif = {
            "n_jobs": Discrete(self.max_n_jobs + 1),
            "n_nodes": Discrete(self.max_n_modes + 1),
            "n_resources": Discrete(self.max_n_resources + 1),
            "n_pr_edges": Discrete(self.max_n_edges + 1),
            "features": Box(
                low=0,
                high=1000,
                shape=(self.max_n_nodes, self.n_features),
            ),
            "pr_edges": Box(
                low=0,
                high=self.max_n_nodes,
                shape=self.shape_pr,
                dtype=np.int64,
            ),
        }

        if self.add_rp_edges != "none":
            dict_specif["n_rp_edges"] = Discrete(self.max_n_edges + 1)
            dict_specif["rp_edges"] = Box(
                low=0,
                high=self.max_n_nodes,
                shape=self.shape_rp,
                dtype=np.int64,
            )
            dict_specif["rp_att"] = Box(
                low=0, high=1000, shape=self.shape_rp_att, dtype=float
            )

        if self.observe_conflicts_as_cliques:
            dict_specif["n_rc_edges"] = Discrete(self.max_n_edges + 1)
            dict_specif["rc_edges"] = Box(
                low=0,
                high=self.max_n_nodes,
                shape=self.shape_rc,
                dtype=np.int64,
            )
            dict_specif["rc_att"] = Box(
                low=0,
                high=1000,
                shape=self.shape_rc_att,
                dtype=np.int64,
            )

        self.observation_space = Dict(dict_specif)

    def get_n_features(self):
        # 3 for task completion times,
        # 1 for is_affected
        # 1 for mandatory selectable
        # 1 for job id (in case of several modes per job)
        # 3 for duration
        # 1 for source/sink/middle
        n_features = 10
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
            f"add resource prcedence edges:       {self.add_rp_edges}\n"
            f"remove old resource info:           {self.remove_old_resource_info}\n"
            f"remove past prec:                   {self.remove_past_prec}\n"
            f"List of features:\n - Task Completion Times\n - selectable\n - duration"
        )

        if len(print_input_list) != 0:
            print(" - " + "\n - ".join(print_input_list) + "\n")
