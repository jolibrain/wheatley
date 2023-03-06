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

from copy import deepcopy
import numpy as np
import torch


def test_init_and_reset(state):
    assert (state.is_affected == 0).all()
    assert (
        state.task_completion_times[:, :, 3]
        == np.array(
            [
                [1, 6, 16, 23, 31],
                [5, 11, 14, 17, 21],
                [4, 8, 12, 16, 20],
                [5, 11, 18, 26, 35],
                [9, 17, 24, 30, 35],
            ]
        )
    ).all()
    assert set(list(state.graph.nodes)) == {i for i in range(25)}
    assert set(list(state.graph.edges)) == {
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (15, 16),
        (16, 17),
        (17, 18),
        (18, 19),
        (20, 21),
        (21, 22),
        (22, 23),
        (23, 24),
    }


def test_done(state):
    assert not state.done()
    state.graph.add_edge(0, 10)
    state.graph.add_edge(10, 17)
    state.graph.add_edge(17, 9)
    state.graph.add_edge(9, 24)
    state.graph.add_edge(11, 16)
    state.graph.add_edge(16, 23)
    state.graph.add_edge(23, 8)
    state.graph.add_edge(8, 3)
    state.graph.add_edge(5, 1)
    state.graph.add_edge(1, 20)
    state.graph.add_edge(20, 12)
    state.graph.add_edge(12, 19)
    state.graph.add_edge(21, 2)
    state.graph.add_edge(2, 7)
    state.graph.add_edge(7, 13)
    state.graph.add_edge(13, 18)
    state.graph.add_edge(15, 6)
    state.graph.add_edge(6, 22)
    state.graph.add_edge(22, 4)
    state.graph.add_edge(6, 14)
    assert not state.done()
    state.graph.add_edge(4, 14)
    assert state.done()


def test_get_machine_node_ids(state):
    assert set(state._get_machine_node_ids(2)) == {1, 5, 12, 19, 20}
    assert set(state._get_machine_node_ids(4)) == {4, 6, 14, 15, 22}


def test_update_completion_times(state):
    state.task_completion_times[:, :, 3][0, 0] = 2
    state._update_completion_times(1)
    assert (
        state.task_completion_times[:, :, 3]
        == np.array(
            [
                [2, 7, 17, 24, 32],
                [5, 11, 14, 17, 21],
                [4, 8, 12, 16, 20],
                [5, 11, 18, 26, 35],
                [9, 17, 24, 30, 35],
            ]
        )
    ).all()


def test_set_precedency(state):
    first_cum_sum = deepcopy(state.task_completion_times[:, :, 3])
    assert state.set_precedency(4, 0) is False
    assert state.set_precedency(0, 4)
    assert state.set_precedency(0, 0) is False
    assert (first_cum_sum == state.task_completion_times[:, :, 3]).all()
    assert state.set_precedency(5, 1)
    assert list(state.task_completion_times[:, :, 3][0]) == [1, 10, 20, 27, 35]
    assert state.set_precedency(5, 20)
    assert list(state.task_completion_times[:, :, 3][4]) == [14, 22, 29, 35, 40]
    assert state.set_precedency(1, 20)
    assert list(state.task_completion_times[:, :, 3][4]) == [19, 27, 34, 40, 45]


def test_get_machine_occupancy(state):
    for i in range(5):
        assert state.get_machine_occupancy(i, "averagistic") == []
    state.affect_node(0)
    state.set_precedency(0, 10)
    state.affect_node(10)
    assert state.get_machine_occupancy(0, "averagistic") == [(0, 1, 0), (1, 4, 10)]
    state.set_precedency(10, 9)
    state.affect_node(9)
    assert state.get_machine_occupancy(0, "averagistic") == [
        (0, 1, 0),
        (1, 4, 10),
        (17, 4, 9),
    ]
    state.set_precedency(24, 9)
    state.affect_node(24)
    assert state.get_machine_occupancy(0, "averagistic") == [
        (0, 1, 0),
        (1, 4, 10),
        (30, 5, 24),
        (35, 4, 9),
    ]


def test_get_solution(state):
    assert state.get_solution() is False
    state.affect_node(0)
    assert state.set_precedency(0, 10)
    state.affect_node(10)
    assert state.set_precedency(10, 17)
    state.affect_node(17)
    assert state.set_precedency(17, 9)
    state.affect_node(9)
    assert state.set_precedency(9, 24)
    state.affect_node(24)
    state.affect_node(11)
    assert state.set_precedency(11, 16)
    state.affect_node(16)
    assert state.set_precedency(16, 23)
    state.affect_node(23)
    assert state.set_precedency(23, 8)
    state.affect_node(8)
    assert state.set_precedency(8, 3)
    state.affect_node(3)
    state.affect_node(5)
    assert state.set_precedency(5, 1)
    state.affect_node(1)
    assert state.set_precedency(1, 20)
    state.affect_node(20)
    assert state.set_precedency(20, 12)
    state.affect_node(12)
    assert state.set_precedency(12, 19)
    state.affect_node(19)
    state.affect_node(21)
    assert state.set_precedency(21, 2)
    state.affect_node(2)
    assert state.set_precedency(2, 7)
    state.affect_node(7)
    assert state.set_precedency(7, 13)
    state.affect_node(13)
    assert state.set_precedency(13, 18)
    state.affect_node(18)
    state.affect_node(15)
    assert state.set_precedency(15, 6)
    state.affect_node(6)
    assert state.set_precedency(6, 22)
    state.affect_node(22)
    assert state.set_precedency(22, 4)
    state.affect_node(4)
    assert state.set_precedency(4, 14)
    state.affect_node(14)
    assert (
        state.get_solution().schedule
        == np.array(
            [
                [0, 5, 27, 43, 50],
                [0, 5, 37, 40, 43],
                [1, 5, 19, 40, 58],
                [0, 9, 15, 44, 52],
                [10, 19, 27, 34, 47],
            ]
        )
    ).all()


def test_get_first_unaffected_task(state):
    assert state.get_first_unaffected_task(0) == 0
    state.affect_node(0)
    state.affect_node(1)
    assert state.get_first_unaffected_task(0) == 2
    state.affect_node(2)
    state.affect_node(3)
    state.affect_node(4)
    assert state.get_first_unaffected_task(0) == -1


def test_get_job_availability(state):
    assert state.get_job_availability(2, 0, "averagsitic") == 0
    assert state.get_job_availability(2, 2, "averagistic") == 8
