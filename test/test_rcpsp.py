import sys

sys.path.append(".")

import pytest
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from utils.rcpsp import Rcpsp
from utils.loaders import PSPLoader

def test_rcpsp_toy():
    n_jobs=3
    n_modes_per_job=[1,1,1]
    durations=[[[2]],[[2]],[[2]]]
    resource_cons = [[[0,1]],[[0,1]],[[2,0]]]
    resource_availabilities = [1,2]
    n_renewable_resources = 2
    successors = [[2],[3],[]]
    rcpsp1 = Rcpsp(n_jobs=n_jobs, n_modes_per_job=n_modes_per_job, successors=successors, durations=durations, resource_cons=resource_cons, resource_availabilities = resource_availabilities, n_renewable_resources = n_renewable_resources)

    assert(rcpsp1.n_jobs == n_jobs)
    assert(rcpsp1.n_modes_per_job == n_modes_per_job)
    assert(rcpsp1.durations == durations)
    assert(rcpsp1.resource_cons == resource_cons)
    assert(rcpsp1.resource_availabilities == resource_availabilities)
    assert(rcpsp1.n_renewable_resources == n_renewable_resources)
    assert(rcpsp1.n_nonrenewable_resources == 0)
    assert(rcpsp1.n_doubly_constrained_resources == 0)
    assert(rcpsp1.n_resources == 2)
    assert(rcpsp1.successors == successors)
    assert(rcpsp1.max_resource_availability == 2)
    assert(rcpsp1.max_resource_consumption == 2)
    assert(rcpsp1.source_id == rcpsp1.job_to_id(4))
    assert(rcpsp1.sink_id == rcpsp1.job_to_id(5))

    dist_source = rcpsp1.computeDistSourceStart()
    assert(dist_source[rcpsp1.job_to_id(1)] == 0)
    assert(dist_source[rcpsp1.job_to_id(2)] == 2)
    assert(dist_source[rcpsp1.job_to_id(3)] == 4)

    dist_sink = rcpsp1.computeDistStartSink()
    assert(dist_sink[rcpsp1.job_to_id(1)] == 6)
    assert(dist_sink[rcpsp1.job_to_id(2)] == 4)
    assert(dist_sink[rcpsp1.job_to_id(3)] == 2)


def test_rcpsp_small():
    psp = PSPLoader()
    rcpsp2 = psp.load_single_rcpsp("instances/psp/small/small.sm")
    assert(rcpsp2.n_jobs == 8)
    assert(rcpsp2.n_modes_per_job == [1,1,1,1,1,1,1,1])
    assert(rcpsp2.durations == [[[0]],[[3]],[[4]],[[6]],[[2]],[[1]],[[4]],[[0]]])
    assert(rcpsp2.resource_cons == [[[0,0]],[[2,1]],[[2,1]],[[2,1]],[[2,1]],[[3,0]],[[3,1]],[[0,0]]])
    assert(rcpsp2.resource_availabilities == [4,2])
    assert(rcpsp2.n_renewable_resources == 2)
    assert(rcpsp2.n_nonrenewable_resources == 0)
    assert(rcpsp2.n_doubly_constrained_resources == 0)
    assert(rcpsp2.n_resources == 2)
    assert(rcpsp2.successors == [[2,3],[4],[4,5],[6],[6,7],[8],[8],[]])
    assert(rcpsp2.max_resource_availability == 4)
    assert(rcpsp2.max_resource_consumption == 3)
    assert(rcpsp2.source_id == rcpsp2.job_to_id(1))
    assert(rcpsp2.sink_id == rcpsp2.job_to_id(8))

    # g = rcpsp2.precGraph
    # nx.draw(g, with_labels = True)
    # plt.show()

    dist_source = rcpsp2.computeDistSourceStart()
    assert(dist_source[rcpsp2.job_to_id(1)] == 0)
    assert(dist_source[rcpsp2.job_to_id(2)] == 0)
    assert(dist_source[rcpsp2.job_to_id(3)] == 0)
    assert(dist_source[rcpsp2.job_to_id(4)] == 4)
    assert(dist_source[rcpsp2.job_to_id(5)] == 4)
    assert(dist_source[rcpsp2.job_to_id(6)] == 10)
    assert(dist_source[rcpsp2.job_to_id(7)] == 6)
    assert(dist_source[rcpsp2.job_to_id(8)] == 11)

    dist_sink = rcpsp2.computeDistStartSink()
    assert(dist_sink[rcpsp2.job_to_id(1)] == 11)
    assert(dist_sink[rcpsp2.job_to_id(2)] == 10)
    assert(dist_sink[rcpsp2.job_to_id(3)] == 11)
    assert(dist_sink[rcpsp2.job_to_id(4)] == 7)
    assert(dist_sink[rcpsp2.job_to_id(5)] == 6)
    assert(dist_sink[rcpsp2.job_to_id(6)] == 1)
    assert(dist_sink[rcpsp2.job_to_id(7)] == 4)
    assert(dist_sink[rcpsp2.job_to_id(8)] == 0)

    rs_1_3 = rcpsp2.sample_topological(1,3)
    rs_2_3 = rcpsp2.sample_topological(2,3)
    rs_2_4 = rcpsp2.sample_topological(2,4)
    rs_1_5 = rcpsp2.sample_topological(1,5)
    rs_1_6 = rcpsp2.sample_topological(1,6)
    rs_3_7 = rcpsp2.sample_topological(3,7)

    assert(rs_1_3.n_jobs == 6)
    assert(rs_1_3.successors_id == [[1, 2], [3], [3, 4], [5], [5], []])
    assert(rs_2_3.n_jobs == 4)
    assert(rs_2_3.successors_id == [[1, 2], [3], [3], []])
    assert(rs_2_3.durations == [[[0]], [[6]], [[2]], [[0]]])
    assert(rs_2_4.n_jobs == 6)
    assert(rs_2_4.successors_id == [[1, 2], [3], [3, 4], [5], [5], []])
    assert(rs_1_5.n_jobs == 8)
    assert(rs_1_5.successors_id == rcpsp2.successors_id)
    assert(rs_1_6.n_jobs == 8)
    assert(rs_1_6.successors_id == rcpsp2.successors_id)
    assert(rs_3_7.n_jobs == 4)
    assert(rs_3_7.successors_id == [[1, 2], [3], [3], []])


def test_rcpsp_c154_3():
    psp = PSPLoader()
    rcpsp3 = psp.load_single_rcpsp("instances/psp/mm/c154_3.mm")
    assert(rcpsp3.n_jobs == 18)
    assert(rcpsp3.n_modes_per_job == [1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,1])
    assert(rcpsp3.durations[rcpsp3.job_to_id(1)] == [[0]])
    assert(rcpsp3.durations[rcpsp3.job_to_id(2)] == [[1],[1],[9]])
    assert(rcpsp3.durations[rcpsp3.job_to_id(3)] == [[4],[5],[7]])
    assert(rcpsp3.durations[rcpsp3.job_to_id(4)] == [[1],[3],[4]])
    assert(rcpsp3.durations[rcpsp3.job_to_id(18)] == [[0]])
    assert(len(rcpsp3.durations) == 18)
    assert(rcpsp3.resource_cons[rcpsp3.job_to_id(1)][rcpsp3.mode_to_id(1)] == [0,0,0,0])
    assert(rcpsp3.resource_cons[rcpsp3.job_to_id(2)][rcpsp3.mode_to_id(1)] == [6,0,0,1])
    assert(rcpsp3.resource_cons[rcpsp3.job_to_id(2)][rcpsp3.mode_to_id(2)] == [0,10,8,0])
    assert(rcpsp3.resource_cons[rcpsp3.job_to_id(2)][rcpsp3.mode_to_id(3)] == [0,8,8,0])
    assert(rcpsp3.resource_cons[rcpsp3.job_to_id(3)][rcpsp3.mode_to_id(1)] == [6,0,0,5])
    assert(rcpsp3.resource_cons[rcpsp3.job_to_id(18)][rcpsp3.mode_to_id(1)] == [0,0,0,0])
    assert(len(rcpsp3.resource_cons) == 18)
    assert(rcpsp3.resource_availabilities == [20,26,23,36])
    assert(rcpsp3.n_renewable_resources == 2)
    assert(rcpsp3.n_nonrenewable_resources == 2)
    assert(rcpsp3.n_doubly_constrained_resources == 0)
    assert(rcpsp3.n_resources == 4)
    assert(rcpsp3.successors == [[2,3,4],[5,10],[8,9,16],[5,7,11],[6,14],
        [12],[9],[10],[15,17],[11,13],[15],[16],[17],[16],[18],[18],[18],[]])
    assert(rcpsp3.max_resource_availability == 36)
    assert(rcpsp3.max_resource_consumption == 10)
    assert(rcpsp3.source_id == rcpsp3.job_to_id(1))
    assert(rcpsp3.sink_id == rcpsp3.job_to_id(18))



    