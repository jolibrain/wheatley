import numpy as np

from utils.utils import generate_problem


def test_generate_problem():
    affectations, durations = generate_problem(5, 5, 0, 10)
    assert (np.sum(affectations, axis=1) == 10 * np.ones(5)).all()
