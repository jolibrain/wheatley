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
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


def decode_mask(info_mask):
    """Add padding to the given list of masks.

    The padding is set to False, which means that this extra pad
    is masked as well.
    """
    if isinstance(info_mask, np.ndarray):
        info_mask = [row for row in info_mask]

    masks = []
    for mask_ in info_mask:
        arr = np.asarray(mask_, dtype=bool)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        masks.append(arr)

    has_multi_agent = any(mask.shape[0] > 1 for mask in masks)
    max_nodes = max(mask.shape[-1] for mask in masks)

    if not has_multi_agent:
        padded = [
            np.concatenate(
                (mask.squeeze(0), np.zeros(max_nodes - mask.shape[-1], dtype=bool))
            )
            for mask in masks
        ]
        return np.stack(padded)

    max_agents = max(mask.shape[0] for mask in masks)
    padded = []
    for mask in masks:
        pad = np.zeros((max_agents, max_nodes), dtype=bool)
        pad[: mask.shape[0], : mask.shape[-1]] = mask
        padded.append(pad)
    return np.stack(padded)


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr:
    :return:
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def get_path(arg_path, exp_name):
    path = os.path.join(arg_path, exp_name)
    if not path.endswith("/"):
        path += "/"

    try:
        os.mkdir(path)
    except OSError as error:
        print("save directory", path, " already exists")
    return path


def logcosh(source, target, reduction):
    err = (source - target).cosh().log()
    if reduction == "none":
        return err
    if reduction == "mean":
        return err.mean()
