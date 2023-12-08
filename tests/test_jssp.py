import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from args import argument_parser, parse_args
from generic.agent_specification import AgentSpecification
from jssp.description import Description
from jssp.env.env import Env
from jssp.env.env_specification import EnvSpecification
from jssp.env.state import State
from jssp.eval import list_instances
from jssp.models.agent import Agent
from jssp.models.custom_agent import CustomAgent
from jssp.solve import solve_instance
from jssp.train import main
from jssp.utils.loaders import load_problem

# This is the list of all experiments we want to try.
# Each entry should be read as "argument_name: [value_experiment_1, value_experiment_2, ...]"
# If an entry has less experiment values than the maximum number of experiments,
# then its last value will be repeated for the missing experiment values.
possible_args = {
    "n_j": [6, 10],
    "n_m": [10, 5],
    "dont_normalize_input": [True, False],
    "sample_n_jobs": [-1, 3],
    "chunk_n_jobs": [3, -1],
    "max_n_j": [20],
    "max_n_m": [20],
    "lr": [0.0001],
    "ent_coef": [0.05],
    "vf_coef": [1.0],
    "clip_range": [0.25],
    "gamma": [1.0],
    "gae_lambda": [1.0],
    "optimizer": ["adamw"],
    "fe_type": ["dgl"],
    "residual_gnn": [True],
    "graph_has_relu": [True],
    "graph_pooling": ["learn", "learninv"],
    "hidden_dim_features_extractor": [32],
    "n_layers_features_extractor": [3],
    "mlp_act": ["gelu"],
    "layer_pooling": ["last"],
    "n_mlp_layers_features_extractor": [1],
    "n_mlp_layers_actor": [1],
    "n_mlp_layers_critic": [1],
    "hidden_dim_actor": [16],
    "hidden_dim_critic": [16],
    "total_timesteps": [1000],
    "n_validation_env": [10],
    "n_steps_episode": [490],
    "batch_size": [245],
    "n_epochs": [1],
    "fixed_validation": [True],
    "custom_heuristic_names": ["SPT MWKR MOPNR FDD/MWKR", "SPT"],
    "seed": [0],
    "duration_type": ["stochastic", "deterministic"],
    "generate_duration_bounds": ["0.05 0.1"],
    "ortools_strategy": ["averagistic realistic", "pessimistic"],
    "device": ["cuda:0"],
    "n_workers": [2],
    "skip_initial_eval": [True, False],
    "return_based_scaling": [True, False],
}

# Duplicate each entry to match the maximum number of possibilities to try.
max_number_of_possibilities = max(len(v) for v in possible_args.values())
possible_args = {
    k: v + (max_number_of_possibilities - len(v)) * [v[-1]]
    for k, v in possible_args.items()
}

# Build the list of all experiments to launch.
# An experiment is defined by a list of arguments preformatted
# for argparse.
args_to_test = [[] for _ in range(max_number_of_possibilities)]
for key, values in possible_args.items():
    for test_id, value in enumerate(values):
        args = args_to_test[test_id]

        if value is True:
            args.append(f"--{key}")
        elif value is False:
            pass
        else:
            args.append(f"--{key}")
            if isinstance(value, str):
                for sub_v in value.split(" "):
                    args.append(f"{sub_v}")
                    print(sub_v)
            else:
                args.append(f"{value}")


@pytest.mark.parametrize(
    "args",
    args_to_test,
)
def test_args(args: list):
    """Make sure the main training function don't crash when using multiple different
    args.
    """
    original_argv = sys.argv

    try:
        # Simulate the arguments passed as input for the argument parser to work seemlessly.
        args.append("--disable_visdom")
        if len(original_argv) == 3:
            output_directory = original_argv[2]
            output_directory += "/" if not output_directory.endswith("/") else ""
            args.append("--path")
            args.append(output_directory)

        sys.argv = ["python3"] + args

        parser = argument_parser()
        args, exp_name, path = parse_args(parser)
        main(args, exp_name, path)
    finally:
        # Don't forget to bring back the old argv!
        sys.argv = original_argv


def test_resets():
    """Test the env resets.

    - Soft reset: change nothing.
    - Hard reset and not fixed problem: change everything.
    - Hard reset and fixed problem: sample new real durations only (stochastic).
    """

    def init_specs(deterministic: bool, fixed: bool):
        env_specification = EnvSpecification(
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
        problem_description = Description(
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
        env = Env(problem_description, env_specification)
        affectations = env.state.affectations.copy()
        durations = env.state.original_durations.copy()

        env.reset(soft=True)
        assert np.all(affectations == env.state.affectations), "Affectations changed"
        assert np.all(durations == env.state.original_durations), "Durations changed"

    # Not fixed problems & hard resets.
    for deterministic in [True, False]:
        env_specification, problem_description = init_specs(deterministic, False)
        env = Env(problem_description, env_specification)
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
        env = Env(problem_description, env_specification)
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
def test_seed(seed: int):
    """Make sure the seed is completely defining the generated problem."""

    def init_specs(deterministic: bool, fixed: bool, seed: int):
        env_specification = EnvSpecification(
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
        problem_description = Description(
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
        env = Env(problem_description, env_specification)
        affectations = env.state.affectations.copy()
        durations = env.state.original_durations.copy()

        env_specification, problem_description = init_specs(deterministic, fixed, seed)
        env = Env(problem_description, env_specification)
        assert np.all(env.state.affectations == affectations), "Affectations changed"
        assert np.all(env.state.original_durations == durations), "Durations changed"

        env_specification, problem_description = init_specs(
            deterministic, fixed, seed + 1
        )
        env = Env(problem_description, env_specification)
        assert np.any(env.state.affectations != affectations), "Affectations unchanged"
        assert np.any(env.state.original_durations != durations), "Durations unchanged"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_state_instance_file(seed: int):
    """
    - Generate a random problem.
    - Save the state on disk.
    - Instantiate a new state from the instance file on disk.
    - Compare the two states to make sure they're the same.
    """

    def init_specs(deterministic: bool, fixed: bool, seed: int):
        env_specification = EnvSpecification(
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
        problem_description = Description(
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
        state_1 = Env(problem_description, env_specification).state
        state_1.save_instance_file(Path(sys.argv[2]) / "state.npz")
        state_2 = State.from_instance_file(
            Path(sys.argv[2]) / "state.npz",
            max_n_jobs=10,
            max_n_machines=10,
            n_features=state_1.features.shape[1],
            deterministic=deterministic,
        )

        assert np.all(state_1.original_durations == state_2.original_durations)
        assert np.all(state_1.affectations == state_2.affectations)


@pytest.mark.parametrize(
    "n_j,n_m,duration_type,expected_results",
    [
        (
            6,
            6,
            "stochastic",
            {
                "SPT": 742.84,
                "MWKR": 699.7,
                "MOPNR": 699.16,
                "FDD/MWKR": 705.88,
            },
        ),
        (
            6,
            6,
            "deterministic",
            {
                "SPT": 551.63,
                "MWKR": 544.79,
                "MOPNR": 545.89,
                "FDD/MWKR": 550.06,
            },
        ),
    ],
)
def test_validation_results(
    n_j: int, n_m: int, duration_type: str, expected_results: dict
):
    instance_dir = Path(f"./instances/jssp/{duration_type}/{n_j}x{n_m}")
    assert instance_dir.is_dir(), f"Validation instances {instance_dir} does not exists"

    instances = list_instances(Path(f"./instances/jssp/{duration_type}"))[(n_j, n_m)]
    agents_solutions = defaultdict(list)
    for instance_file in instances:
        state = State.from_instance_file(
            instance_file,
            n_j,
            n_m,
            n_features=6 + n_m,
            deterministic=duration_type == "deterministic",
        )

        for rule in expected_results.keys():
            agent = CustomAgent(rule, "averagistic")
            solution = agent.predict(state.original_durations, state.affectations)
            agents_solutions[rule].append(solution.get_makespan())

    for rule, solutions in agents_solutions.items():
        assert expected_results[rule] == np.mean(
            solutions
        ), f"Different solutions found ({expected_results[rule]} vs {np.mean(solutions)})"


@pytest.mark.parametrize(
    "instance_file,taillard_offset",
    [
        ("./instances/taillard/ta01.txt", True),
        ("./instances/test/4x4_sparse.txt", False),
    ],
)
def test_solve_api(instance_file: str, taillard_offset: bool):
    n_j, n_m, affectations, durations = load_problem(
        instance_file,
        taillard_offset,
        deterministic=True,
    )
    env_specification = EnvSpecification(
        max_n_jobs=n_j,
        max_n_machines=n_m,
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
    agent_specification = AgentSpecification(
        env_specification.n_features,
        gconv_type="gatv2",
        graph_has_relu=True,
        graph_pooling="max",
        layer_pooling="last",
        mlp_act="gelu",
        mlp_act_graph="gelu",
        device="cuda",
        n_mlp_layers_features_extractor=1,
        n_layers_features_extractor=1,
        hidden_dim_features_extractor=32,
        n_attention_heads=1,
        reverse_adj=False,
        residual_gnn=True,
        normalize_gnn=True,
        conflicts="clique",
        n_mlp_layers_actor=1,
        hidden_dim_actor=16,
        n_mlp_layers_critic=1,
        hidden_dim_critic=16,
        fe_type="dgl",
        transformer_flavor="linear",
        dropout=0.1,
        cache_lap_node_id=True,
        lap_node_id_k=10,
        edge_embedding_flavor="sum",
        performer_nb_features=None,
        performer_feature_redraw_interval=1000,
        performer_generalized_attention=False,
        performer_auto_check_redraw=False,
        vnode=False,
        update_edge_features=True,
        update_edge_features_pe=True,
        ortho_embed=False,
        no_tct=False,
        mid_in_edges=False,
        rwpe_k=0,
        rwpe_h=16,
        cache_rwpe=False,
        two_hot=None,
        symlog=False,
    )

    agent = Agent(env_specification, agent_specification=agent_specification)
    solution = solve_instance(agent, affectations, durations, deterministic=True)
    assert solution is not None
