from copy import deepcopy

import networkx as nx
import numpy as np


def test_get_mask(l2d_transition_model, state):
    for i in range(25):
        if i in [0, 5, 10, 15, 20]:
            assert l2d_transition_model.get_mask(state)[i] == 1
        else:
            assert l2d_transition_model.get_mask(state)[i] == 0
    state.affect_node(5)
    state.affect_node(6)
    for i in range(25):
        if i in [0, 7, 10, 15, 20]:
            assert l2d_transition_model.get_mask(state)[i] == 1
        else:
            assert l2d_transition_model.get_mask(state)[i] == 0


def test_run(l2d_transition_model, state):

    backup_state = deepcopy(state)

    # Without forced insertion
    l2d_tm = deepcopy(l2d_transition_model)
    state = deepcopy(backup_state)
    state_cp = deepcopy(backup_state)

    # Check actions that change only affectations
    l2d_tm.run(state, 5, False)
    l2d_tm.run(state, 6, False)
    l2d_tm.run(state, 0, False)
    assert state.is_affected[1, 0] == 1
    assert state.is_affected[1, 1] == 1
    assert state.is_affected[0, 0] == 1
    assert nx.is_isomorphic(state.graph, state_cp.graph)
    assert (state.task_completion_times == state_cp.task_completion_times).all()

    # Check actions that change graph and affectations (and btw, insertion before)
    l2d_tm.run(state, 15, False)
    assert state.is_affected[3, 0] == 1
    assert state.graph.has_edge(15, 6)
    assert (state.task_completion_times == state_cp.task_completion_times).all()

    # Check actions that change graph, affectations and task_completion_times
    l2d_tm.run(state, 20, False)
    l2d_tm.run(state, 10, False)
    l2d_tm.run(state, 1, False)
    assert state.is_affected[4, 0] == 1
    assert state.is_affected[2, 0] == 1
    assert state.is_affected[0, 1] == 1
    assert state.graph.has_edge(5, 20)
    assert state.graph.has_edge(0, 10)
    assert state.graph.has_edge(20, 1)
    assert (
        state.task_completion_times[:, :, 3]
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
    l2d_tm.run(state, 7, False)
    l2d_tm.run(state, 2, False)
    assert state.is_affected[1, 2] == 1
    assert state.is_affected[0, 2] == 1
    assert state.graph.has_edge(7, 2)
    assert (
        state.task_completion_times[:, :, 3]
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
    l2d_tm.run(state, 21, False)
    assert state.is_affected[4, 1] == 1
    assert state.graph.has_edge(7, 2)
    assert state.graph.has_edge(2, 21)
    assert not state.graph.has_edge(7, 21)
    assert not state.graph.has_edge(21, 2)
    assert (
        state.task_completion_times[:, :, 3]
        == np.array(
            [
                [1, 19, 29, 36, 44],
                [5, 11, 14, 17, 21],
                [5, 9, 13, 17, 21],
                [5, 11, 18, 26, 35],
                [14, 37, 44, 50, 55],
            ]
        )
    ).all()

    # With forced insertion
    l2d_tm = deepcopy(l2d_transition_model)
    state = deepcopy(backup_state)
    state_cp = deepcopy(backup_state)

    # Check if unallowed actions have no effects, as they should
    l2d_tm.run(state_cp, 50, True)
    assert eq(state, state_cp)
    l2d_tm.run(state_cp, 1, True)
    assert eq(state, state_cp)

    # Check actions that change only affectations
    l2d_tm.run(state, 5, True)
    l2d_tm.run(state, 6, True)
    l2d_tm.run(state, 0, True)
    assert state.is_affected[1, 0] == 1
    assert state.is_affected[1, 1] == 1
    assert state.is_affected[0, 0] == 1
    assert nx.is_isomorphic(state.graph, state_cp.graph)
    assert (state.task_completion_times == state_cp.task_completion_times).all()

    # Check actions that change graph and affectations (and btw, insertion before)
    l2d_tm.run(state, 15, True)
    assert state.is_affected[3, 0] == 1
    assert state.graph.has_edge(15, 6)
    assert (state.task_completion_times == state_cp.task_completion_times).all()

    # Check actions that change graph, affectations and task_completion_times
    l2d_tm.run(state, 20, True)
    l2d_tm.run(state, 10, True)
    l2d_tm.run(state, 1, True)
    assert state.is_affected[4, 0] == 1
    assert state.is_affected[2, 0] == 1
    assert state.is_affected[0, 1] == 1
    assert state.graph.has_edge(5, 20)
    assert state.graph.has_edge(0, 10)
    assert state.graph.has_edge(20, 1)
    assert (
        state.task_completion_times[:, :, 3]
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
    l2d_tm.run(state, 7, True)
    l2d_tm.run(state, 2, True)
    assert state.is_affected[1, 2] == 1
    assert state.is_affected[0, 2] == 1
    assert state.graph.has_edge(7, 2)
    assert (
        state.task_completion_times[:, :, 3]
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
    l2d_tm.run(state, 21, True)
    assert state.is_affected[4, 1] == 1
    assert state.graph.has_edge(7, 21)
    assert state.graph.has_edge(21, 2)
    assert not state.graph.has_edge(7, 2)
    assert (
        state.task_completion_times[:, :, 3]
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
        and (state1.task_completion_times == state2.task_completion_times).all()
        and (state1.is_affected == state2.is_affected).all()
    ):
        return True
    return False
