import numpy as np

from env.l2d_transition_model import L2DTransitionModel, MachineOccupancy
from config import MAX_N_NODES


def test_get_next_job_task_id():
    mo = MachineOccupancy(3, np.array([[0, 1, 2], [2, 1, 0], [0, 2, 1]]))
    task_id, machine = mo.get_next_job_task_id(1, 0)
    assert task_id == -1


def test_other():
    pass
    # TODO : write the rest of the tests for l2d_transition_model
