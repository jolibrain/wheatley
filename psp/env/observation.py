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
import numpy as np


class EnvObservation:
    def __init__(
        self,
        env_specification,
        n_jobs,
        n_modes,
        n_resources,
        max_n_jobs,
        max_n_modes,
        max_n_resources,
        max_edges_factor,
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges=None,
        resource_prec_att=None,
    ):
        """
        This should only hanlde cpu tensors, since it is used on the env side.
        """

        if resource_conf_edges is None:
            self.observe_conflicts_as_cliques = False
        else:
            self.observe_conflicts_as_cliques = True
        self.n_jobs = n_jobs
        self.n_modes = n_modes
        self.n_resources = n_resources
        self.max_n_jobs = max_n_jobs
        self.max_n_modes = max_n_modes
        self.max_n_resources = max_n_resources
        # if max_edges_factor > 0:
        #     self.max_n_pb_edges = self.max_n_modes * max_edges_factor
        #     self.max_n_rc_edges = (
        #         2 * self.max_n_modes * max_edges_factor * self.max_n_resources
        #     )
        #     self.max_n_rp_edges = (
        #         self.max_n_modes * max_edges_factor * self.max_n_resources * 4
        #     )

        # else:
        #     self.max_n_pb_edges = self.max_n_modes**2
        #     self.max_n_rc_edges = self.max_n_modes**2 * self.max_n_resources
        #     self.max_n_rp_edges = self.max_n_modes**2
        self.n_nodes = n_modes

        self.features = features
        self.problem_edge_index = problem_edge_index
        self.resource_conf_edges = resource_conf_edges
        self.resource_conf_att = resource_conf_att
        self.resource_prec_edges = resource_prec_edges
        self.resource_prec_att = resource_prec_att
        self.env_specification = env_specification
        self.add_rp_edges = env_specification.add_rp_edges

    def get_n_nodes(self):
        return self.n_nodes

    def get_n_rp_edges(self):
        if self.resource_prec_edges is None:
            return 0
        return self.resource_prec_edges.shape[1]

    def get_n_pr_edges(self):
        if not self.problem_edge_index.any():
            return 0
        return self.problem_edge_index.shape[1]

    def get_n_rc_edges(self):
        if self.resource_conf_edges is None:
            return 0
        return self.resource_conf_edges.shape[1]

    def to_gym_observation(self):
        features = np.zeros(
            (self.max_n_modes, self.features.shape[1]), dtype=np.float32
        )
        features[: self.features.shape[0], :] = self.features
        pr_edge_index = np.zeros(self.env_specification.shape_pr, dtype=np.int64)
        pr_edge_index[:, : self.get_n_pr_edges()] = self.problem_edge_index

        ret = {
            "n_jobs": self.n_jobs,
            "n_nodes": self.n_modes,
            "n_resources": self.n_resources,
            "n_pr_edges": self.get_n_pr_edges(),
            "n_rp_edges": self.get_n_rp_edges(),
            "features": features,
            "pr_edges": pr_edge_index,
        }

        if self.add_rp_edges != "none":
            rp_edge_index = np.zeros(self.env_specification.shape_rp, dtype=np.int64)
            rp_att = np.zeros(self.env_specification.shape_rp_att, dtype=np.float32)
            if self.resource_prec_edges is not None:
                rp_edge_index[:, : self.get_n_rp_edges()] = self.resource_prec_edges
                rp_att[: self.get_n_rp_edges(), :] = self.resource_prec_att
            ret["n_rp_edges"] = self.get_n_rp_edges()
            ret["rp_edges"] = rp_edge_index
            ret["rp_att"] = rp_att

        if self.observe_conflicts_as_cliques:
            rc_edge_index = np.zeros(self.env_specification.shape_rc, dtype=np.int64)
            rc_edge_index[:, : self.get_n_rc_edges()] = self.resource_conf_edges
            rc_att = np.zeros(self.env_specification.shape_rc_att, dtype=np.float32)
            rc_att[: self.get_n_rc_edges(), :] = self.resource_conf_att

            ret["n_rc_edges"] = self.get_n_rc_edges()
            ret["rc_edges"] = rc_edge_index
            ret["rc_att"] = rc_att

        return ret
