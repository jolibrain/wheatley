from copy import deepcopy

import networkx as nx
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
    l2d_tm_cp.run(102)
    # Check if unallowed actions have no effects, as they should
    assert eq(l2d_tm.state, l2d_tm_cp.state)
    l2d_tm_cp.run(5050)
    assert eq(l2d_tm.state, l2d_tm_cp.state)
    l2d_tm_cp.run(202)
    assert eq(l2d_tm.state, l2d_tm_cp.state)

    l2d_tm.run(505)
    assert l2d_tm.state.is_affected[1, 0] == 1
    l2d_tm.run(606)
    l2d_tm.run(0)
    assert l2d_tm.state.is_affected[1, 1] == 1
    assert l2d_tm.state.is_affected[0, 0] == 1
    assert nx.is_isomorphic(l2d_tm.state.graph, l2d_tm_cp.state.graph)
    assert (
        l2d_tm.state.task_completion_times
        == l2d_tm_cp.state.task_completion_times
    ).all()

    l2d_tm.run(1515)
    assert l2d_tm.state.is_affected[3, 0] == 1
    assert l2d_tm.state.task_completion_times[3, 0] == 5
    assert l2d_tm.state.task_completion_times[1, 1] == 11

    l2d_tm.run(2222)
    assert l2d_tm.state.task_completion_times[4, 2] == 24

    l2d_tm.run(1414)
    assert l2d_tm.state.task_completion_times[2, 4] == 20
    assert l2d_tm.state.task_completion_times[4, 2] == 27

    l2d_tm.run(404)
    assert l2d_tm.state.task_completion_times[0, 4] == 36


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
