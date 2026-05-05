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


class EnvSpecification:
    def __init__(
        self,
        normalize_input,
        max_n_nodes,
    ):
        self.normalize_input = normalize_input
        self.n_features = self.get_n_features()
        self.max_n_nodes = max_n_nodes
        self.max_n_steps = max_n_nodes  ##XXX(beniz): to give more exploration in MA
        # self.max_n_steps = max_n_nodes / 2
        # self.max_n_steps = 6
        # self.max_n_steps = (
        #     max_n_nodes * 10
        # )  ##XXX(beniz): to give more exploration in MA

    def get_n_features(self):
        # 2 for coordinates,
        n_features = 2
        return n_features

    def print_self(self):
        print(
            f"==========Env Description     ==========\n"
            f"Input normalization:                {'Yes' if self.normalize_input else 'No'}\n"
        )
