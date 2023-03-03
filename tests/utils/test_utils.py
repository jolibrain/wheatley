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

from utils.utils import node_to_job_and_task, job_and_task_to_node


def test_node_to_job_and_task():
    assert node_to_job_and_task(13, 5) == (2, 3)
    assert node_to_job_and_task(9, 5) == (1, 4)
    assert node_to_job_and_task(0, 1) == (0, 0)


def test_job_and_task_to_node():
    assert job_and_task_to_node(0, 4, 5) == 4
    assert job_and_task_to_node(5, 5, 6) == 35
    assert job_and_task_to_node(3, 2, 4) == 14
