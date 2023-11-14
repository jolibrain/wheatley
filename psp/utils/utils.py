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
import os
import sys
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


def compute_resources_graph_np(r_info):
    n_modes = r_info.shape[0]
    # r_info is n_modes x n_resources, contains values between 0 and 1
    conflicts = np.where(
        np.logical_and(
            np.expand_dims(r_info, 0) != 0,
            np.expand_dims(r_info, 1) != 0,
            out=np.zeros(
                (r_info.shape[0], r_info.shape[0], r_info.shape[1]), dtype=bool
            ),
            where=np.expand_dims(np.logical_not(np.diag([True] * n_modes)), 2),
        )
    )
    # conflicts[0] is source of edge
    # conflicts[1] is dest of edge
    # conflicts[2] is ressource id
    # both directions are created at once
    conflicts_val = r_info[conflicts[0], conflicts[2]]
    conflicts_val_r = r_info[conflicts[1], conflicts[2]]
    return (
        np.stack([conflicts[0], conflicts[1]]),
        conflicts[2],
        conflicts_val,
        conflicts_val_r,
    )


def compute_resources_graph_torch(r_info):
    n_modes = r_info.shape[0]
    # r_info is n_modes x n_resources, contains values between 0 and 1
    notdiag = torch.logical_not(
        torch.diag(torch.BoolTensor([True] * n_modes)).unsqueeze_(-1)
    )
    # m1 = r_info.unsqueeze(0) != 0
    # m2 = r_info.unsqueeze(1) != 0

    # c1 = torch.logical_and(m1, m2)
    # c2 = torch.logical_and(c1, notdiag)
    c2 = torch.logical_and(
        torch.logical_and(
            r_info.unsqueeze(0).expand(r_info.shape[0], -1, -1) != 0,
            r_info.unsqueeze(1).expand(-1, r_info.shape[0], -1) != 0,
        ),
        notdiag,
    )
    conflicts = torch.where(c2)
    # conflicts[0] is source of edge
    # conflicts[1] is dest of edge
    # conflicts[2] is ressource id
    # both directions are created at once
    conflicts_val = r_info[conflicts[0], conflicts[2]]
    conflicts_val_r = r_info[conflicts[1], conflicts[2]]
    return (
        torch.stack([conflicts[0], conflicts[1]]),
        conflicts[2],
        conflicts_val,
        conflicts_val_r,
    )
