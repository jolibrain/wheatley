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


class PSPEnvObservation:
    def __init__(
        self,
        n_jobs,
        n_modes,
        features,
        edge_index,
        conflicts_edges,
        conflicts_edges_resourceinfo,
        max_n_jobs,
        max_n_machines,
        max_edges_factor,
    ):
        """
        This should only hanlde cpu tensors, since it is used on the env side.
        """

        if features.is_cuda:
            raise Exception("Please provide a cpu observation")

        if conflicts_edges is None:
            self.observe_conflicts_as_cliques = False
        else:
            self.observe_conflicts_as_cliques = True
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.max_n_jobs = max_n_jobs
        self.max_n_machines = max_n_machines
        self.max_n_nodes = self.max_n_jobs * self.max_n_machines
        if max_edges_factor > 0:
            self.max_n_edges = self.max_n_nodes * max_edges_factor
        else:
            self.max_n_edges = self.max_n_nodes**2
        self.n_nodes = n_jobs * n_machines
        self.n_edges = edge_index.shape[1]

        self.features = features
        self.edge_index = edge_index

        if self.observe_conflicts_as_cliques:
            self.conflicts_edges = conflicts_edges
            self.n_conflict_edges = conflicts_edges.shape[1]
            self.conflicts_edges_machineid = conflicts_edges_machineid

        assert self.n_nodes == self.features.shape[0]

    def get_n_nodes(self):
        return self.n_nodes

    def get_n_edges(self):
        return self.n_edges

    def to_gym_observation(self):
        """
        This should only handle cpu tensors, since it is used on the env side.
        """
        if self.features.is_cuda:
            raise Exception("Please use this only on cpu observation")

        features = torch.empty((self.max_n_nodes, self.features.shape[1]))
        features[0 : self.get_n_nodes(), :] = self.features
        edge_index = np.empty((2, self.max_n_edges))
        edge_index[:, 0 : self.get_n_edges()] = self.edge_index

        if self.observe_conflicts_as_cliques:
            conflicts_edges = torch.empty(
                (2, 2 * sum(range(self.max_n_jobs)) * self.max_n_machines)
            )
            conflicts_edges[:, 0 : self.conflicts_edges.shape[1]] = self.conflicts_edges
            conflicts_edges_machineid = torch.empty(
                (2, 2 * sum(range(self.max_n_jobs)) * self.max_n_machines)
            )
            conflicts_edges_machineid[
                :, : self.conflicts_edges_machineid.shape[0]
            ] = self.conflicts_edges_machineid.unsqueeze(0)

            return {
                "n_jobs": self.n_jobs,
                "n_machines": self.n_machines,
                "n_nodes": self.n_nodes,
                "n_edges": self.n_edges,
                "features": features.numpy(),
                "edge_index": edge_index.astype("int64"),
                "n_conflict_edges": self.n_conflict_edges,
                "conflicts_edges": conflicts_edges.numpy().astype("int64"),
                "conflicts_edges_machineid": conflicts_edges_machineid.numpy().astype(
                    "int64"
                ),
            }
        else:
            return {
                "n_jobs": self.n_jobs,
                "n_machines": self.n_machines,
                "n_nodes": self.n_nodes,
                "n_edges": self.n_edges,
                "features": features.numpy(),
                "edge_index": edge_index.astype("int64"),
            }
