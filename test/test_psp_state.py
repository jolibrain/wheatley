import sys

sys.path.append(".")

import pytest
import numpy as np
from env.psp_state import PSPState
from problem.psp_description import PSPDescription

from env.psp_env_specification import PSPEnvSpecification
from utils.psp_env_observation import PSPEnvObservation as EnvObservation
from utils.psp_agent_observation import PSPAgentObservation as AgentObservation

import torch


def test_state(state_small):
    s = state_small
    # no node is affected
    assert np.all(s.features[:, 0] == 0)
    # only node 0 is selectable
    assert s.features[0, 1] == 1
    assert np.all(s.features[1:, 1] == 0)

    s.affect_job(0)
    # only node 0 is affected
    assert s.features[0, 0] == 1
    assert np.all(s.features[1:, 0] == 0)
    # node 1 and 2 are the only selectable
    assert s.features[0, 1] == 0
    assert s.features[1, 1] == 1
    assert s.features[2, 1] == 1
    assert np.all(s.features[3:, 1] == 0)

    s.affect_job(1)
    assert np.all(s.features[:2, 0] == 1)
    assert np.all(s.features[2:, 0] == 0)
    assert np.all(s.features[:2, 1] == 0)
    assert np.all(s.features[2, 1] == 1)
    assert np.all(s.features[3:, 1] == 0)

    s.affect_job(2)
    assert np.all(s.features[:3, 0] == 1)
    assert np.all(s.features[3:, 0] == 0)
    assert np.all(s.features[:3, 1] == 0)
    assert np.all(s.features[3:5, 1] == 1)
    assert np.all(s.features[5:, 1] == 0)

    s.affect_job(3)
    assert s.tct_real(3) == 10.0

    s.affect_job(4)
    assert s.tct_real(4) == 6.0

    s.affect_job(5)
    assert s.tct_real(5) == 11.0

    s.affect_job(6)
    assert s.tct_real(6) == 15.0

    s.affect_job(7)
    assert s.tct_real(7) == 15.0


def test_rcatt(state_small_preclique):
    sp = state_small_preclique
    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = sp.to_features_and_edge_index(False)

    eogp = EnvObservation(
        sp.env_specification,
        sp.problem["n_jobs"],
        sp.n_nodes,
        sp.problem["n_resources"],
        sp.problem_description.max_n_jobs,
        sp.problem_description.max_n_modes,
        sp.env_specification.max_n_resources,
        sp.env_specification.max_edges_factor,
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ).to_gym_observation()

    eogtp = AgentObservation.np_to_torch(eogp)

    print("building obs before affectations")
    op = AgentObservation.from_gym_observation(eogtp, conflicts="clique")

    gp = op.to_graph()

    sp.affect_job(0)
    sp.affect_job(1)
    sp.affect_job(2)
    sp.affect_job(3)
    sp.affect_job(4)

    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = sp.to_features_and_edge_index(False)

    eogp = EnvObservation(
        sp.env_specification,
        sp.problem["n_jobs"],
        sp.n_nodes,
        sp.problem["n_resources"],
        sp.problem_description.max_n_jobs,
        sp.problem_description.max_n_modes,
        sp.env_specification.max_n_resources,
        sp.env_specification.max_edges_factor,
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ).to_gym_observation()

    eogtp = AgentObservation.np_to_torch(eogp)

    print("building obs after affectations")
    op = AgentObservation.from_gym_observation(eogtp, conflicts="clique")


def test_obs(state_small, state_small_preclique):

    sp = state_small_preclique
    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = sp.to_features_and_edge_index(False)

    eogp = EnvObservation(
        sp.env_specification,
        sp.problem["n_jobs"],
        sp.n_nodes,
        sp.problem["n_resources"],
        sp.problem_description.max_n_jobs,
        sp.problem_description.max_n_modes,
        sp.env_specification.max_n_resources,
        sp.env_specification.max_edges_factor,
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ).to_gym_observation()

    eogtp = AgentObservation.np_to_torch(eogp)

    op = AgentObservation.from_gym_observation(eogtp, conflicts="clique")

    gp = op.to_graph()

    s = state_small
    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = s.to_features_and_edge_index(False)

    eog = EnvObservation(
        s.env_specification,
        s.problem["n_jobs"],
        s.n_nodes,
        s.problem["n_resources"],
        s.problem_description.max_n_jobs,
        s.problem_description.max_n_modes,
        s.env_specification.max_n_resources,
        s.env_specification.max_edges_factor,
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ).to_gym_observation()

    eogt = AgentObservation.np_to_torch(eog)

    o = AgentObservation.from_gym_observation(eogt, conflicts="clique")

    g = o.to_graph()
    assert torch.equal(g.edges()[0], gp.edges()[0])
    ge0 = g.edges()[0].clone()
    ge1 = g.edges()[1].clone()
    assert torch.equal(g.edges()[1], gp.edges()[1])
    assert torch.equal(g.edata["type"], gp.edata["type"])
    # assert torch.equal(g.edata["att_rp"], gp.edata["att_rp"])
    assert torch.equal(g.edata["att_rc"], gp.edata["att_rc"])

    s.affect_job(0)
    s.affect_job(1)
    s.affect_job(2)
    s.affect_job(3)
    s.affect_job(4)
    sp.affect_job(0)
    sp.affect_job(1)
    sp.affect_job(2)
    sp.affect_job(3)
    sp.affect_job(4)

    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = sp.to_features_and_edge_index(False)

    eogp = EnvObservation(
        sp.env_specification,
        sp.problem["n_jobs"],
        sp.n_nodes,
        sp.problem["n_resources"],
        sp.problem_description.max_n_jobs,
        sp.problem_description.max_n_modes,
        sp.env_specification.max_n_resources,
        sp.env_specification.max_edges_factor,
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ).to_gym_observation()

    eogtp = AgentObservation.np_to_torch(eogp)

    op = AgentObservation.from_gym_observation(eogtp, conflicts="clique")

    gp = op.to_graph()

    s = state_small
    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = s.to_features_and_edge_index(False)

    eog = EnvObservation(
        s.env_specification,
        s.problem["n_jobs"],
        s.n_nodes,
        s.problem["n_resources"],
        s.problem_description.max_n_jobs,
        s.problem_description.max_n_modes,
        s.env_specification.max_n_resources,
        s.env_specification.max_edges_factor,
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ).to_gym_observation()

    eogt = AgentObservation.np_to_torch(eog)

    o = AgentObservation.from_gym_observation(eogt, conflicts="clique")

    g = o.to_graph()

    assert torch.equal(g.edges()[0], gp.edges()[0])
    assert torch.equal(g.edges()[1], gp.edges()[1])
    assert torch.equal(g.edata["type"], gp.edata["type"])
    assert torch.equal(g.edata["att_rp"], gp.edata["att_rp"])
    assert torch.equal(g.edata["att_rc"], gp.edata["att_rc"])


def test_graph(state_small):
    s = state_small
    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = s.to_features_and_edge_index(False)

    eog = EnvObservation(
        s.env_specification,
        s.problem["n_jobs"],
        s.n_nodes,
        s.problem["n_resources"],
        s.problem_description.max_n_jobs,
        s.problem_description.max_n_modes,
        s.env_specification.max_n_resources,
        s.env_specification.max_edges_factor,
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ).to_gym_observation()

    eogt = AgentObservation.np_to_torch(eog)

    o = AgentObservation.from_gym_observation(eogt, conflicts="att")

    g = o.to_graph()
    orige0 = torch.LongTensor(
        [
            0,
            0,
            1,
            2,
            2,
            3,
            4,
            4,
            5,
            6,
            1,
            2,
            3,
            3,
            4,
            5,
            5,
            6,
            7,
            7,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ]
    )
    orige1 = torch.LongTensor(
        [
            1,
            2,
            3,
            3,
            4,
            5,
            5,
            6,
            7,
            7,
            0,
            0,
            1,
            2,
            2,
            3,
            4,
            4,
            5,
            6,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ]
    )
    assert torch.equal(g.edges()[0], orige0)
    assert torch.equal(g.edges()[1], orige1)
    assert torch.equal(g.edata["type"], torch.LongTensor([1] * 10 + [2] * 10 + [0] * 8))

    g2 = AgentObservation.from_gym_observation(eogt, conflicts="clique").to_graph()
    e0pb = g.edges()[0][:20]
    e1pb = g.edges()[1][:20]
    e0sl = g.edges()[0][-8:]
    e1sl = g.edges()[1][-8:]
    e0rc = torch.LongTensor([1] * 9 + [2] * 9 + [3] * 9 + [4] * 9 + [5] * 5 + [6] * 9)
    e1rc = torch.LongTensor(
        [
            2,
            2,
            3,
            3,
            4,
            4,
            5,
            6,
            6,
            1,
            1,
            3,
            3,
            4,
            4,
            5,
            6,
            6,
            1,
            1,
            2,
            2,
            4,
            4,
            5,
            6,
            6,
            1,
            1,
            2,
            2,
            3,
            3,
            5,
            6,
            6,
            1,
            2,
            3,
            4,
            6,
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            4,
            5,
        ]
    )

    assert torch.equal(g2.edges()[0], torch.cat([e0pb, e0rc, e0sl]))
    assert torch.equal(g2.edges()[1], torch.cat([e1pb, e1rc, e1sl]))
    assert torch.equal(
        g2.edata["type"],
        torch.LongTensor([1] * 10 + [2] * 10 + [3] * e0rc.shape[0] + [0] * 8),
    )
    # TODO : check edata["att_rc"]


def test_render(state_small):
    s = state_small
    s.affect_job(0)
    s.affect_job(1)
    s.affect_job(2)
    s.affect_job(3)
    s.affect_job(4)
    s.affect_job(5)
    s.affect_job(6)
    s.affect_job(7)
    s.render_solution(s.get_solution().schedule)


def test_state_nonren(state_nonren):
    s = state_nonren

    print("s.job_modes", s.job_modes)
    # ortools solution
    s.affect_job(0)  # j0/0
    print("\n")
    nid = (1 - 1) * 3 + 1 + 0  # j1/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (2 - 1) * 3 + 1 + 2  # j2/2
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (3 - 1) * 3 + 1 + 1  # j3/1
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (4 - 1) * 3 + 1 + 0  # j4/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (5 - 1) * 3 + 1 + 0  # j5/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (6 - 1) * 3 + 1 + 1  # j6/1
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (7 - 1) * 3 + 1 + 1  # j7/1
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (8 - 1) * 3 + 1 + 2  # j8/2
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (9 - 1) * 3 + 1 + 0  # j9/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (10 - 1) * 3 + 1 + 0  # j10/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (11 - 1) * 3 + 1 + 0  # j11/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (12 - 1) * 3 + 1 + 0  # j12/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (13 - 1) * 3 + 1 + 0  # j13/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (14 - 1) * 3 + 1 + 1  # j14/1
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (15 - 1) * 3 + 1 + 1  # j15/1
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (16 - 1) * 3 + 1 + 0  # j16/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (17 - 1) * 3 + 1 + 0  # j17/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)

    makespan = s.tct(-1)[0].item()
    print("makespan", makespan)

    s.reset()
    # WHEATLEY solution
    s.affect_job(0)  # j0/0
    print("\n")
    nid = (1 - 1) * 3 + 1 + 0  # j1/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (2 - 1) * 3 + 1 + 0  # j2/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (3 - 1) * 3 + 1 + 1  # j3/1
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (4 - 1) * 3 + 1 + 0  # j4/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (5 - 1) * 3 + 1 + 0  # j5/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (6 - 1) * 3 + 1 + 0  # j6/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (7 - 1) * 3 + 1 + 0  # j7/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (8 - 1) * 3 + 1 + 0  # j8/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (9 - 1) * 3 + 1 + 0  # j9/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (10 - 1) * 3 + 1 + 0  # j10/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (11 - 1) * 3 + 1 + 1  # j11/1
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (12 - 1) * 3 + 1 + 0  # j12/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (13 - 1) * 3 + 1 + 0  # j13/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (14 - 1) * 3 + 1 + 1  # j14/1
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (15 - 1) * 3 + 1 + 2  # j15/2
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (16 - 1) * 3 + 1 + 0  # j16/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    nid = (17 - 1) * 3 + 1 + 0  # j17/0
    for r in range(2, 4):
        print("r", r)
        print("avail", s.resources[r][0].remaining_level)
        print("rusage", s.resource_usage(nid, r))
        assert s.resource_usage(nid, r) <= s.resources[r][0].remaining_level
    print("affecting", nid)
    s.affect_job(nid)
    makespan = s.tct(-1)[0].item()
    print("makespan", makespan)


def test_forget(state_small):
    s = state_small
    s.forget_nodes()
    print("affecting 0")
    s.affect_job(0)
    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = s.to_features_and_edge_index(True)
    print("features.shape", features.shape)

    print("affecting 1")
    s.affect_job(1)
    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = s.to_features_and_edge_index(True)
    print("features.shape", features.shape)
    print("affecting 2")
    s.affect_job(2)
    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = s.to_features_and_edge_index(True)
    print("features.shape", features.shape)
    print("affecting 3")
    s.affect_job(3)
    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = s.to_features_and_edge_index(True)
    print("features.shape", features.shape)
    print("affecting 4")
    s.affect_job(4)
    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = s.to_features_and_edge_index(True)
    print("features.shape", features.shape)
    print("affecting 5")
    s.affect_job(5)
    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = s.to_features_and_edge_index(True)
    print("features.shape", features.shape)
    print("affecting 6")
    s.affect_job(6)
    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = s.to_features_and_edge_index(True)
    print("features.shape", features.shape)
    print("affecting 7")
    s.affect_job(7)
    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = s.to_features_and_edge_index(True)
    print("features.shape", features.shape)
