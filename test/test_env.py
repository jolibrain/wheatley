from env.psp_env import PSPEnv
import numpy as np


def test_env(problem_description_small, env_specification_small):
    env = PSPEnv(problem_description_small, env_specification_small, 0)
    obs, reward, done, _, info = env.step(0)
    assert done == False
    assert reward == 0
    assert np.all(
        np.equal(
            info["mask"],
            np.array([False, True, True, False, False, False, False, False]),
        )
    )

    obs, reward, done, _, info = env.step(1)
    assert done == False
    assert reward == 0
    assert np.all(
        np.equal(
            info["mask"],
            np.array([False, False, True, False, False, False, False, False]),
        )
    )

    obs, reward, done, _, info = env.step(2)
    assert done == False
    assert reward == 0
    assert np.all(
        np.equal(
            info["mask"],
            np.array([False, False, False, True, True, False, False, False]),
        )
    )

    obs, reward, done, _, info = env.step(3)
    assert done == False
    assert reward == 0
    assert np.all(
        np.equal(
            info["mask"],
            np.array([False, False, False, False, True, False, False, False]),
        )
    )
    obs, reward, done, _, info = env.step(4)
    assert done == False
    assert reward == 0
    assert np.all(
        np.equal(
            info["mask"],
            np.array([False, False, False, False, False, True, True, False]),
        )
    )
    obs, reward, done, _, info = env.step(5)
    assert done == False
    assert reward == 0
    assert np.all(
        np.equal(
            info["mask"],
            np.array([False, False, False, False, False, False, True, False]),
        )
    )
    obs, reward, done, _, info = env.step(6)
    assert done == False
    assert reward == 0
    assert np.all(
        np.equal(
            info["mask"],
            np.array([False, False, False, False, False, False, False, True]),
        )
    )

    obs, reward, done, _, info = env.step(7)
    assert done
    # assert reward == -15 unnormalized
    # assert reward == -1.25 normalized final value
    assert reward == -15 / env.state.max_duration / env.state.n_resources
    assert np.all(
        np.equal(
            info["mask"],
            np.array([False, False, False, False, False, False, False, False]),
        )
    )
