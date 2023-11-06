import sys

sys.path.append(".")

import pytest
import numpy as np
from psp.env.state import State
from psp.description import Description

from psp.env.observation import EnvObservation
from psp.models.agent_observation import AgentObservation

import torch


def test_gstate(gstate_small):
    s = gstate_small
    assert s.graph.ndata["selectable"][0].item() == 1
    assert torch.all(s.graph.ndata["selectable"][1:] == 0)

    s.affect_job(0)
    assert s.affected(0)
    assert torch.all(s.graph.ndata["affected"][1:] == 0)
    # node 1 and 2 are the only selectable
    assert s.graph.ndata["selectable"][0].item() == 0
    assert s.graph.ndata["selectable"][1].item() == 1
    assert s.graph.ndata["selectable"][2].item() == 1
    assert torch.all(s.graph.ndata["selectable"][3:] == 0)

    s.affect_job(1)
    assert s.affected(0)
    assert s.affected(1)
    assert not s.affected(2)
    assert not s.affected(3)
    assert not s.affected(4)
    assert not s.affected(5)
    assert not s.affected(6)
    assert not s.affected(7)
    assert not s.selectable(0)
    assert not s.selectable(1)
    assert s.selectable(2)
    assert not s.selectable(3)
    assert not s.selectable(4)
    assert not s.selectable(5)
    assert not s.selectable(6)
    assert not s.selectable(7)

    s.affect_job(2)
    assert s.affected(0)
    assert s.affected(1)
    assert s.affected(2)
    assert not s.affected(3)
    assert not s.affected(4)
    assert not s.affected(5)
    assert not s.affected(6)
    assert not s.affected(7)
    assert not s.selectable(0)
    assert not s.selectable(1)
    assert not s.selectable(2)
    assert s.selectable(3)
    assert s.selectable(4)
    assert not s.selectable(5)
    assert not s.selectable(6)
    assert not s.selectable(7)

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

    print(s.observe())
    # TODO check resources
