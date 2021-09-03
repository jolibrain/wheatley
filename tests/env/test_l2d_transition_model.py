from copy import deepcopy

import networkx as nx
import numpy as np
import torch


def test_get_mask(l2d_transition_model):
    for i in range(625):
        if i in [0, 130, 260, 390, 520]:
            assert l2d_transition_model.get_mask()[i] == 1
        else:
            assert l2d_transition_model.get_mask()[i] == 0
    l2d_transition_model.state.affect_node(5)
    l2d_transition_model.state.affect_node(6)
    for i in range(625):
        if i in [0, 182, 260, 390, 520]:
            assert l2d_transition_model.get_mask()[i] == 1
        else:
            assert l2d_transition_model.get_mask()[i] == 0


def test_run(l2d_transition_model):
    l2d_tm = l2d_transition_model
    l2d_tm_cp = deepcopy(l2d_tm)

    # Check if unallowed actions have no effects, as they should
    l2d_tm_cp.run(0, 1)
    assert eq(l2d_tm.state, l2d_tm_cp.state)
    l2d_tm_cp.run(50, 50)
    assert eq(l2d_tm.state, l2d_tm_cp.state)
    l2d_tm_cp.run(1, 1)
    assert eq(l2d_tm.state, l2d_tm_cp.state)

    # Check actions that change only affectations
    l2d_tm.run(5, 5)
    l2d_tm.run(6, 6)
    l2d_tm.run(0, 0)
    assert l2d_tm.state.is_affected[1, 0] == 1
    assert l2d_tm.state.is_affected[1, 1] == 1
    assert l2d_tm.state.is_affected[0, 0] == 1
    assert nx.is_isomorphic(l2d_tm.state.graph, l2d_tm_cp.state.graph)
    assert (
        l2d_tm.state.task_completion_times
        == l2d_tm_cp.state.task_completion_times
    ).all()

    # Check actions that change graph and affectations (and btw, insertion before)
    l2d_tm.run(15, 15)
    assert l2d_tm.state.is_affected[3, 0] == 1
    assert l2d_tm.state.graph.has_edge(15, 6)
    assert (
        l2d_tm.state.task_completion_times
        == l2d_tm_cp.state.task_completion_times
    ).all()

    # Check actions that change graph, affectations and task_completion_times
    l2d_tm.run(20, 20)
    l2d_tm.run(10, 10)
    l2d_tm.run(1, 1)
    assert l2d_tm.state.is_affected[4, 0] == 1
    assert l2d_tm.state.is_affected[2, 0] == 1
    assert l2d_tm.state.is_affected[0, 1] == 1
    assert l2d_tm.state.graph.has_edge(5, 20)
    assert l2d_tm.state.graph.has_edge(0, 10)
    assert l2d_tm.state.graph.has_edge(20, 1)
    assert (
        l2d_tm.state.task_completion_times
        == np.array(
            [
                [1, 19, 29, 36, 44],
                [5, 11, 14, 17, 21],
                [5, 9, 13, 17, 21],
                [5, 11, 18, 26, 35],
                [14, 22, 29, 35, 40],
            ]
        )
    ).all()

    # Check insertion between 2 nodes
    l2d_tm.run(7, 7)
    l2d_tm.run(2, 2)
    assert l2d_tm.state.is_affected[1, 2] == 1
    assert l2d_tm.state.is_affected[0, 2] == 1
    assert l2d_tm.state.graph.has_edge(7, 2)
    assert (
        l2d_tm.state.task_completion_times
        == np.array(
            [
                [1, 19, 29, 36, 44],
                [5, 11, 14, 17, 21],
                [5, 9, 13, 17, 21],
                [5, 11, 18, 26, 35],
                [14, 22, 29, 35, 40],
            ]
        )
    ).all()
    l2d_tm.run(21, 21)
    assert l2d_tm.state.is_affected[4, 1] == 1
    assert l2d_tm.state.graph.has_edge(7, 21)
    assert l2d_tm.state.graph.has_edge(21, 2)
    assert l2d_tm.state.graph.has_edge(7, 2)
    assert (
        l2d_tm.state.task_completion_times
        == np.array(
            [
                [1, 19, 32, 39, 47],
                [5, 11, 14, 17, 21],
                [5, 9, 13, 17, 21],
                [5, 11, 18, 26, 35],
                [14, 22, 29, 35, 40],
            ]
        )
    ).all()


def eq(state1, state2):
    if (
        nx.is_isomorphic(state1.graph, state2.graph)
        and (
            state1.task_completion_times == state2.task_completion_times
        ).all()
        and (state1.is_affected == state2.is_affected).all()
    ):
        return True
    return False
