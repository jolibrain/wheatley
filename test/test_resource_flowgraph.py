import sys

sys.path.append(".")

# import pytest
from utils.resource_flowgraph import ResourceFlowGraph


def test_resource_flowgraph():
    rg = ResourceFlowGraph(4, renewable=True)
    a0 = rg.availability(3)
    assert a0 == 0
    assert rg.frontier == [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    assert rg.find_max_pos(0) == 3
    rg.consume(1, 3, 0, 4)
    assert rg.frontier == [[0, 0, 1], [4, 1, 1], [4, 1, 1], [4, 1, 1]]

    a1 = rg.availability(1)
    assert a1 == 0
    assert rg.find_max_pos(0) == 0
    rg.consume(2, 1, 0, 6)
    assert rg.frontier == [[4, 1, 1], [4, 1, 1], [4, 1, 1], [6, 2, 1]]

    a2 = rg.availability(2)
    assert a2 == 4
    assert rg.find_max_pos(4) == 2
    rg.consume(3, 2, 4, 9)
    assert rg.frontier == [[4, 1, 1], [6, 2, 1], [9, 3, 1], [9, 3, 1]]

    a3 = rg.availability(4)
    assert a3 == 9
    assert rg.find_max_pos(11) == 3
    rg.consume(4, 4, 11, 17)
    assert rg.frontier == [[17, 4, 1], [17, 4, 1], [17, 4, 1], [17, 4, 1]]


def test_resource_flowgraph_cumul():
    rg = ResourceFlowGraph(4, renewable=True)
    rg.consume(1, 2, 0, 3)
    rg.consume(2, 2, 0, 4)
    rg.consume(3, 2, 4, 10)
    rg.consume(4, 2, 4, 6)
    rg.consume(5, 3, 10, 11)
    rg.consume(6, 3, 11, 15)
    assert rg.edges == [(0, 1), (0, 2), (2, 3), (1, 4), (3, 5), (4, 5), (5, 6)]
    assert rg.edges_att == [2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 3.0]

    rg2 = ResourceFlowGraph(4, renewable=True)
    rg2.consume(1, 1, 0, 3)
    rg2.consume(2, 1, 0, 4)
    rg2.consume(3, 1, 4, 10)
    rg2.consume(4, 1, 4, 6)
    rg2.consume(6, 1, 11, 15)
    assert rg2.edges == [(0, 1), (0, 2), (2, 3), (1, 4), (3, 6)]
    assert rg2.edges_att == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_resource_flowgraph_norm():
    rg = ResourceFlowGraph(1, unit_val=0.25, renewable=True)
    date = rg.availability(0.5)
    assert date == 0
    rg.consume(1, 0.5, 0, 3)
    date2 = rg.availability(0.5)
    assert date2 == 0
    rg.consume(2, 0.5, 0, 4)
    date3 = rg.availability(0.5)
    assert date3 == 3
    rg.consume(3, 0.5, 4, 10)
    date4 = rg.availability(0.5)
    assert date4 == 3
    rg.consume(4, 0.5, 4, 6)
    date5 = rg.availability(0.75)
    assert date5 == 10
    rg.consume(5, 0.75, 10, 11)
    date6 = rg.availability(0.75)
    assert date6 == 11
    rg.consume(6, 0.75, 11, 15)
    assert rg.edges == [(0, 1), (0, 2), (2, 3), (1, 4), (3, 5), (4, 5), (5, 6)]
    assert rg.edges_att == [0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.75]


def test_resource_flowgraph_insert():
    rg = ResourceFlowGraph(10, renewable=True)
    rg.consume(1, 5, 0, 3)
    rg.consume(2, 8, 3, 6)
    date = rg.availability(5)
    assert date == 6
    date2 = rg.availability(2)
    assert date2 == 0


def test_resource_flowgraph_insert_norm():
    rg = ResourceFlowGraph(1, unit_val=0.1, renewable=True)
    rg.consume(1, 0.5, 0, 3)
    rg.consume(2, 0.8, 3, 6)
    date = rg.availability(0.5)
    assert date == 6
    date2 = rg.availability(0.2)
    assert date2 == 0
