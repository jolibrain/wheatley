from itertools import product

import numpy as np
import pytest

from env.jssp_env import JSSPEnv
from env.jssp_env_specification import JSSPEnvSpecification
from problem.jssp_description import JSSPDescription


def test_resets():
    """Test the env resets.

    - Soft reset: change nothing.
    - Hard reset and not fixed problem: change everything.
    - Hard reset and fixed problem: sample new real durations only (stochastic).
    """

    def init_specs(deterministic: bool, fixed: bool):
        env_specification = JSSPEnvSpecification(
            max_n_jobs=10,
            max_n_machines=10,
            normalize_input=True,
            input_list=["duration"],
            insertion_mode="no_forced_insertion",
            max_edges_factor=4,
            sample_n_jobs=-1,
            chunk_n_jobs=-1,
            observe_conflicts_as_cliques=True,
            observe_real_duration_when_affect=False,
            do_not_observe_updated_bounds=False,
        )
        problem_description = JSSPDescription(
            deterministic=deterministic,
            fixed=fixed,
            transition_model_config="simple",
            reward_model_config="Sparse",
            duration_mode_bounds=(10, 50),
            duration_delta=(10, 200),
            n_jobs=10,
            n_machines=10,
            max_duration=99,
            seed=0,
        )
        return env_specification, problem_description

    # Test the soft scenarios.
    for deterministic, fixed in product([True, False], [True, False]):
        env_specification, problem_description = init_specs(deterministic, fixed)
        env = JSSPEnv(problem_description, env_specification)
        affectations = env.state.affectations.copy()
        durations = env.state.original_durations.copy()

        env.reset(soft=True)
        assert np.all(affectations == env.state.affectations), "Affectations changed"
        assert np.all(durations == env.state.original_durations), "Durations changed"

    # Not fixed problems & hard resets.
    for deterministic in [True, False]:
        env_specification, problem_description = init_specs(deterministic, False)
        env = JSSPEnv(problem_description, env_specification)
        affectations = env.state.affectations.copy()
        durations = env.state.original_durations.copy()

        env.reset(soft=False)
        assert np.any(affectations != env.state.affectations), "Affectations unchanged"
        assert np.any(
            durations[:, :, 1:] != env.state.original_durations[:, :, 1:]
        ), "Bornes unchanged"
        assert np.any(
            durations[:, :, 0] != env.state.original_durations[:, :, 0]
        ), "Real durations unchanged"

    # Fixed problems and hard resets.
    for deterministic in [True, False]:
        env_specification, problem_description = init_specs(deterministic, True)
        env = JSSPEnv(problem_description, env_specification)
        affectations = env.state.affectations.copy()
        durations = env.state.original_durations.copy()

        env.reset(soft=False)
        assert np.any(affectations == env.state.affectations), "Affectations unchanged"
        assert np.any(
            durations[:, :, 1:] == env.state.original_durations[:, :, 1:]
        ), "Bornes unchanged"
        if deterministic:
            assert np.any(
                durations[:, :, 0] == env.state.original_durations[:, :, 0]
            ), "Real durations changed"
        else:
            assert np.any(
                durations[:, :, 0] != env.state.original_durations[:, :, 0]
            ), "Real durations unchanged"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_problem_generation(seed: int):
    """Make sure the seed is completely defining the generated problem."""

    def init_specs(deterministic: bool, fixed: bool, seed: int):
        env_specification = JSSPEnvSpecification(
            max_n_jobs=10,
            max_n_machines=10,
            normalize_input=True,
            input_list=["duration"],
            insertion_mode="no_forced_insertion",
            max_edges_factor=4,
            sample_n_jobs=-1,
            chunk_n_jobs=-1,
            observe_conflicts_as_cliques=True,
            observe_real_duration_when_affect=False,
            do_not_observe_updated_bounds=False,
        )
        problem_description = JSSPDescription(
            deterministic=deterministic,
            fixed=fixed,
            transition_model_config="simple",
            reward_model_config="Sparse",
            duration_mode_bounds=(10, 50),
            duration_delta=(10, 200),
            n_jobs=10,
            n_machines=10,
            max_duration=99,
            seed=seed,
        )
        return env_specification, problem_description

    for deterministic, fixed in product([False, True], [False, True]):
        env_specification, problem_description = init_specs(deterministic, fixed, seed)
        env = JSSPEnv(problem_description, env_specification)
        affectations = env.state.affectations.copy()
        durations = env.state.original_durations.copy()

        env_specification, problem_description = init_specs(deterministic, fixed, seed)
        env = JSSPEnv(problem_description, env_specification)
        assert np.all(env.state.affectations == affectations), "Affectations changed"
        assert np.all(env.state.original_durations == durations), "Durations changed"

        env_specification, problem_description = init_specs(deterministic, fixed, seed + 1)
        env = JSSPEnv(problem_description, env_specification)
        assert np.any(env.state.affectations != affectations), "Affectations unchanged"
        assert np.any(env.state.original_durations != durations), "Durations unchanged"
